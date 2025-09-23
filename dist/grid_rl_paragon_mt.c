// grid_rl_paragon_mt.c — Grid RL via Paragon + multithread model search
// Build:
//   gcc grid_rl_paragon_mt.c -I. -L. -lparacast -lncursesw -lm -pthread \
//       -Wl,-rpath,'$ORIGIN' -o grid_rl_paragon_mt
//
// Key features:
// - Stable Q: small rewards + tanh head
// - Invalid action masking (no more banging into walls forever)
// - ε-greedy + Boltzmann (softmax) exploration
// - Randomised training starts; fixed demo start (--start y,x)
// - Multithread CPU search: --workers N trains N models in parallel, picks best
// - GPU supported only with --workers 1
//
// CLI:
//   --gpu                use GPU (1 worker only)
//   --workers N          number of CPU workers (default 2; ignored if --gpu)
//   --episodes K         train episodes per worker (default 1500)
//   --demoeps M          demo episodes to evaluate/save (default 6)
//   --start y,x          demo start position (default 4,0)
//   --save file.json     save best model
//   --load file.json     load and demo only
//   --debug              verbose errors to screen

#include <locale.h>
#include <ncursesw/ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <pthread.h>

#include "libparacast.h"

// ===== Grid =====
#define ROWS 5
#define COLS 5
#define ACTIONS 4
#define NUM_STATES (ROWS*COLS)

static int grid[ROWS][COLS] = {
    {0,0,0,1,0},
    {0,1,0,1,0},
    {0,0,0,0,2}, // Goal [2][4]
    {0,1,0,1,0},
    {0,0,0,0,0}  // Start default [4][0]
};
static int action_deltas[ACTIONS][2] = {{-1,0},{1,0},{0,-1},{0,1}}; // U D L R
static char action_chars[ACTIONS] = {'U','D','L','R'};

// ===== Colors =====
#define C_WALL 1
#define C_GOAL 2
#define C_AGENT 3
#define C_INFO 4
#define C_BOLD 5
#define C_ERR  6

// ===== Hyperparams (global defaults; workers sample variations) =====
#define MAX_STEPS 60
#define DEMO_EPS_DEFAULT 6
#define EPISODES_DEFAULT 1500
#define EPS_START_DEFAULT 0.9f
#define EPS_END_DEFAULT   0.05f
#define GAMMA_DEFAULT     0.95f
#define LR_DEFAULT        0.0007f
#define EPOCHS_PER_UPD    1
#define BATCH_SZ          1

// Reward scaling for stability + tanh head
#define R_GOAL      1.0f
#define R_BUMP     -0.25f
#define R_STEP     -0.01f
#define R_PROGRESS  0.02f

static int DEMO_EPS = DEMO_EPS_DEFAULT;
static int TRAIN_EPISODES = EPISODES_DEFAULT;
static int START_Y = 4, START_X = 0;   // demo start (fixed)
static const int GOAL_Y = 2, GOAL_X = 4;

static int debug_mode = 0;

// For optional global locking of Paracast calls (paranoid mode)
static int enable_global_lock = 0;
static pthread_mutex_t paracast_mutex = PTHREAD_MUTEX_INITIALIZER;
#define PC_LOCK()   do{ if(enable_global_lock) pthread_mutex_lock(&paracast_mutex); }while(0)
#define PC_UNLOCK() do{ if(enable_global_lock) pthread_mutex_unlock(&paracast_mutex); }while(0)

// ===== Utils =====
static inline int in_bounds(int y,int x){
    return (x>=0 && x<COLS && y>=0 && y<ROWS && grid[y][x]!=1);
}
static inline int is_goal(int y,int x){ return grid[y][x]==2; }
static inline int get_state(int y,int x){ return y*COLS + x; }
static inline int manhattan(int y,int x,int yy,int xx){
    int dy = y>yy? y-yy: yy-y; int dx = x>xx? x-xx: xx-x; return dy+dx;
}

static void encode_state(int s, float *buf){
    for(int i=0;i<NUM_STATES;i++) buf[i]=0.0f;
    if(s>=0 && s<NUM_STATES) buf[s]=1.0f;
}

// Environment step with shaping (small, positive when closer)
static float step_env(int y,int x,int a,int *ny,int *nx){
    int before = manhattan(y,x,GOAL_Y,GOAL_X);
    int dy = action_deltas[a][0], dx = action_deltas[a][1];
    *ny = y+dy; *nx = x+dx;

    if(!in_bounds(*ny,*nx)){ *ny=y; *nx=x; return R_BUMP + R_STEP; }
    if(is_goal(*ny,*nx)) return R_GOAL;
    int after = manhattan(*ny,*nx,GOAL_Y,GOAL_X);
    float shape = (after<before)? R_PROGRESS : 0.0f;
    return R_STEP + shape;
}

// ===== Paragon glue =====
static int forward_q(long long net, const float *state, float *q4){
    int ok;
    PC_LOCK();
    ok = Paracast_Forward(net, (float*)state, NUM_STATES, 1);
    if(ok){ ok = (Paracast_GetOutput(net, q4, ACTIONS)==ACTIONS); }
    PC_UNLOCK();
    return ok;
}

static int train_tgt(long long net, const float *state, const float *target, float lr, int use_gpu){
    int ok;
    PC_LOCK();
    ok = Paracast_Train(net,
        (float*)state, NUM_STATES, 1,
        (float*)target, ACTIONS, 1,
        BATCH_SZ,
        EPOCHS_PER_UPD,
        lr,
        use_gpu?1:0,
        -0.5f, 0.5f   // tight clip
    );
    PC_UNLOCK();
    return ok;
}

static long long make_net(int use_gpu, unsigned int seed){
    int layers[] = { NUM_STATES,1,  64,1,  32,1,  ACTIONS,1 };
    const char* acts[] = { "linear","relu","relu","tanh" }; // bounded head
    unsigned char fully[] = {1,1,1,1};
    long long net;

    PC_LOCK();
    net = Paracast_NewNetwork(layers, 4, (char**)acts, 4, fully, 4, seed);
    if(net && use_gpu){
        if(!Paracast_InitGPU(net) && debug_mode){
            const char* err = Paracast_GetLastError();
            fprintf(stderr,"GPU init failed (CPU fallback): %s\n", err?err:"(nil)");
        }
    }
    PC_UNLOCK();
    return net;
}

static long long load_net(const char* path, int use_gpu){
    FILE* f = fopen(path,"rb");
    if(!f) return 0;
    fseek(f,0,SEEK_END); long len=ftell(f); fseek(f,0,SEEK_SET);
    char* buf = (char*)malloc(len+1); if(!buf){ fclose(f); return 0; }
    fread(buf,1,len,f); buf[len]=0; fclose(f);
    long long net;

    PC_LOCK();
    net = Paracast_LoadFromJSON(buf, use_gpu?1:0);
    PC_UNLOCK();

    free(buf);
    return net;
}

static int save_net(long long net, const char* path){
    int ok;
    PC_LOCK();
    ok = Paracast_SaveJSON(net, (char*)path);
    PC_UNLOCK();
    return ok;
}

// ===== Action selection with masking & mixed exploration =====
static void mask_invalid_q(int y,int x, float *q){
    // If moving would be invalid, push that Q way down before argmax/softmax
    for(int a=0;a<ACTIONS;a++){
        int ny=y+action_deltas[a][0], nx=x+action_deltas[a][1];
        if(!in_bounds(ny,nx)){
            q[a] = -1e9f;
        }
    }
}
static int argmax4(const float *q){
    int b=0; for(int i=1;i<4;i++) if(q[i]>q[b]) b=i; return b;
}
static int boltzmann_sample(const float *q, float temp){
    // temp>0; softmax over q/temp
    float z=0, p[4]; for(int i=0;i<4;i++){ p[i]=expf(q[i]/fmaxf(temp,1e-3f)); z+=p[i]; }
    if(z<=0){ return rand()%4; }
    float r = ((float)rand()/RAND_MAX)*z;
    float acc=0;
    for(int i=0;i<4;i++){ acc+=p[i]; if(r<=acc) return i; }
    return 3;
}

// ===== Drawing (single-thread, main thread only for demo) =====
static void draw_grid(int ax,int ay, const float *q){
    clear();
    attron(COLOR_PAIR(C_INFO));
    mvprintw(0,0,"+----- Grid (Paragon Q) -----+");
    for(int i=0;i<ROWS;i++){ mvprintw(1+i,0,"|"); mvprintw(1+i,COLS+1,"|"); }
    mvprintw(ROWS+1,0,"+---------------------------+");
    attroff(COLOR_PAIR(C_INFO));

    for(int y=0;y<ROWS;y++){
        for(int x=0;x<COLS;x++){
            int col=1+x;
            if(y==ay && x==ax){ attron(COLOR_PAIR(C_AGENT)); mvprintw(1+y,col,"@"); attroff(COLOR_PAIR(C_AGENT)); }
            else if(grid[y][x]==1){ attron(COLOR_PAIR(C_WALL)); mvprintw(1+y,col,"#"); attroff(COLOR_PAIR(C_WALL)); }
            else if(grid[y][x]==2){ attron(COLOR_PAIR(C_GOAL)); mvprintw(1+y,col,"G"); attroff(COLOR_PAIR(C_GOAL)); }
            else { mvprintw(1+y,col,"."); }
        }
    }
    if(q){
        mvprintw(ROWS+2,0,"Q(U D L R): ");
        float m = fmaxf(fmaxf(q[0],q[1]), fmaxf(q[2],q[3]));
        for(int i=0;i<4;i++){
            int bars = (m>1e-6f)? (int)roundf((q[i]/m)*10.f) : 0;
            attron(COLOR_PAIR(C_INFO)); for(int b=0;b<bars;b++) printw("="); attroff(COLOR_PAIR(C_INFO));
            printw(" %.2f ", q[i]);
        }
        printw(" | Best: %c", action_chars[argmax4(q)]);
    }
    refresh();
}

// ===== Episode runner =====
typedef struct {
    // hyperparams per worker
    unsigned int seed;
    float eps_start, eps_end;
    float boltz_temp;     // for softmax branch
    float lr;
    int episodes;
    int use_gpu;
    // outputs
    long long net;        // trained net
    float eval_avg_reward;
    int   eval_successes;
} WorkerCfg;

static float run_episode(long long net, int train_mode,
                         int *steps_out, char *logbuf, int logmax,
                         float eps, float temp, float lr, int use_gpu,
                         int start_y, int start_x, int viz)
{
    int y=start_y, x=start_x;
    float total=0.f;
    int steps=0;
    float s_vec[NUM_STATES], q[4];

    if(viz){ draw_grid(x,y,NULL); usleep(120000); }

    while(steps<MAX_STEPS){
        int s = get_state(y,x);
        encode_state(s, s_vec);

        int a;
        if(train_mode && ((float)rand()/RAND_MAX) < eps){
            // exploration branch: mix random and Boltzmann 50/50
            if(((float)rand()/RAND_MAX) < 0.5f){
                a = rand()%ACTIONS;
            }else{
                if(!forward_q(net, s_vec, q)){ a = rand()%ACTIONS; }
                else{ mask_invalid_q(y,x,q); a = boltzmann_sample(q, temp); }
            }
        }else{
            if(!forward_q(net, s_vec, q)){ a = rand()%ACTIONS; }
            else{ mask_invalid_q(y,x,q); a = argmax4(q); }
        }

        if(logbuf && steps<logmax-1){ logbuf[steps]=action_chars[a]; logbuf[steps+1]=0; }

        int ny,nx; float r = step_env(y,x,a,&ny,&nx);
        int ns = get_state(ny,nx);
        total += r;

        if(train_mode){
            // TD(0) target
            float ns_vec[NUM_STATES], q_s[4], q_ns[4];
            encode_state(s, s_vec);
            encode_state(ns, ns_vec);
            // forward (again) for targets – tiny extra cost
            if(!forward_q(net, s_vec, q_s)) break;
            if(!forward_q(net, ns_vec, q_ns)) break;
            float maxn = q_ns[0]; for(int i=1;i<4;i++) if(q_ns[i]>maxn) maxn = q_ns[i];
            float target[4] = { q_s[0], q_s[1], q_s[2], q_s[3] };
            float td = (r==R_GOAL) ? r : (r + GAMMA_DEFAULT*maxn);
            // clamp to tanh range
            if(td> 1.0f) td= 1.0f; if(td<-1.0f) td=-1.0f;
            target[a] = td;
            if(!train_tgt(net, s_vec, target, lr, use_gpu)) break;
        }

        y=ny; x=nx; steps++;
        if(viz){
            float q_here[4];
            if(forward_q(net, s_vec, q_here)){ mask_invalid_q(y,x,q_here); draw_grid(x,y, q_here); }
            else { draw_grid(x,y, NULL); }
            usleep(train_mode?80000:140000);
        }

        if(r==R_GOAL) break;               // success ends episode
        if(!train_mode && r<=R_BUMP+R_STEP) break; // demo ends on bump
    }

    if(steps_out) *steps_out = steps;
    return total;
}

// ===== Worker thread: train + eval =====
typedef struct {
    WorkerCfg cfg;
} ThreadArg;

static float eval_model(long long net, int episodes, int start_y, int start_x, int use_gpu){
    float total=0.f; int succ=0;
    for(int ep=0; ep<episodes; ep++){
        int steps; char path[64]={0};
        float r = run_episode(net, 0, &steps, path, 64, 0,1.0f, 0.0f, use_gpu, start_y, start_x, 0);
        total += r;
        if(r>=0.8f) succ++; // treat near-1 as success
    }
    return total/(float)episodes; // success count filled by caller if needed
}

static void* worker_main(void* argp){
    ThreadArg* A = (ThreadArg*)argp;
    WorkerCfg* C = &A->cfg;

    srand(C->seed);

    long long net = make_net(C->use_gpu, C->seed);
    if(!net){ A->cfg.net=0; pthread_exit(NULL); }

    float eps = C->eps_start;
    float eps_decay = powf(C->eps_end/C->eps_start, 1.0f/(float)C->episodes);

    // Training: randomise start each episode for robustness
    for(int ep=0; ep<C->episodes; ep++){
        // pick any free cell that isn't a wall/goal
        int y,x;
        do{ y = rand()%ROWS; x = rand()%COLS; } while(!in_bounds(y,x) || is_goal(y,x));
        int steps; char path[1]={0};
        (void)run_episode(net, 1, &steps, path, 1, eps, C->boltz_temp, C->lr, C->use_gpu, y, x, 0);
        eps = fmaxf(C->eps_end, eps*eps_decay);
    }

    // Evaluate from the demo start
    int succ=0; float total=0.f;
    for(int ep=0; ep<DEMO_EPS; ep++){
        int steps; char path[64]={0};
        float r = run_episode(net, 0, &steps, path, 64, 0, 1.0f, 0.0f, C->use_gpu, START_Y, START_X, 0);
        total += r;
        if(r>=0.8f) succ++;
    }

    C->net = net;
    C->eval_avg_reward = total/(float)DEMO_EPS;
    C->eval_successes = succ;
    pthread_exit(NULL);
}

// ===== CLI / Main =====
static void parse_start(const char* s){
    int y,x;
    if(sscanf(s,"%d,%d",&y,&x)==2){
        if(y>=0 && y<ROWS && x>=0 && x<COLS && in_bounds(y,x)){
            START_Y=y; START_X=x;
        }
    }
}

int main(int argc,char**argv){
    // CLI
    int use_gpu=0;
    int workers=2;
    const char* save_path=NULL;
    const char* load_path=NULL;

    static struct option opts[]={
        {"gpu", no_argument, 0,'g'},
        {"workers", required_argument, 0,'w'},
        {"episodes", required_argument, 0,'e'},
        {"demoeps", required_argument, 0,'n'},
        {"start", required_argument, 0,'S'},
        {"save", required_argument, 0,'s'},
        {"load", required_argument, 0,'l'},
        {"debug", no_argument, 0,'D'},
        {"lock", no_argument, 0,'L'},
        {0,0,0,0}
    };
    int c;
    while((c=getopt_long(argc,argv,"gw:e:n:S:s:l:DL",opts,NULL))!=-1){
        if(c=='g') use_gpu=1;
        else if(c=='w') workers=atoi(optarg);
        else if(c=='e') TRAIN_EPISODES=atoi(optarg);
        else if(c=='n') DEMO_EPS=atoi(optarg);
        else if(c=='S') parse_start(optarg);
        else if(c=='s') save_path=optarg;
        else if(c=='l') load_path=optarg;
        else if(c=='D') debug_mode=1;
        else if(c=='L') enable_global_lock=1;
    }
    if(use_gpu && workers>1){ workers=1; }

    // UI (for demo / status)
    setlocale(LC_ALL,"");
    initscr(); noecho(); curs_set(0);
    start_color();
    init_pair(C_WALL,  COLOR_RED,    COLOR_BLACK);
    init_pair(C_GOAL,  COLOR_GREEN,  COLOR_BLACK);
    init_pair(C_AGENT, COLOR_CYAN,   COLOR_BLACK);
    init_pair(C_INFO,  COLOR_YELLOW, COLOR_BLACK);
    init_pair(C_BOLD,  COLOR_WHITE,  COLOR_BLACK);
    init_pair(C_ERR,   COLOR_MAGENTA,COLOR_BLACK);

    // If loading a model: just load and demo
    if(load_path){
        long long net = load_net(load_path, use_gpu);
        if(!net){ endwin(); fprintf(stderr,"Failed to load %s\n", load_path); return 1; }

        attron(COLOR_PAIR(C_BOLD));
        mvprintw(0,0,"DEMO (loaded): %s | Start: %d,%d | GPU:%s  ", load_path, START_Y, START_X, use_gpu?"on":"off");
        attroff(COLOR_PAIR(C_BOLD));
        refresh();

        float total=0.f; int steps_sum=0, succ=0;
        for(int ep=0; ep<DEMO_EPS; ep++){
            int steps; char path[64]={0};
            float r = run_episode(net, 0, &steps, path, 64, 0,1.0f, 0.0f, use_gpu, START_Y, START_X, 1);
            total+=r; steps_sum+=steps; if(r>=0.8f) succ++;
            attron(COLOR_PAIR(C_INFO));
            mvprintw(ROWS+4+ep,0,"Ep %d: R %6.2f | Steps %3d | Path: %-20s", ep, r, steps, path);
            attroff(COLOR_PAIR(C_INFO)); refresh();
        }
        float avg=total/DEMO_EPS;
        attron(COLOR_PAIR(C_BOLD));
        mvprintw(ROWS+4+DEMO_EPS,0,"Avg: %6.2f | Steps/Ep: %3d | Success: %d/%d",
                 avg, steps_sum/DEMO_EPS, succ, DEMO_EPS);
        attroff(COLOR_PAIR(C_BOLD));
        getch(); endwin(); Paracast_Free(net); return 0;
    }

    // Multi-worker training
    attron(COLOR_PAIR(C_BOLD));
    mvprintw(0,0,"Training search: workers=%d, episodes=%d, demo=%d, GPU:%s  ",
             workers, TRAIN_EPISODES, DEMO_EPS, use_gpu?"on":"off");
    attroff(COLOR_PAIR(C_BOLD));
    refresh();

    pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t)*workers);
    ThreadArg*  ta = (ThreadArg*) malloc(sizeof(ThreadArg)*workers);

    // Sample hyperparams per worker
    unsigned int base_seed = (unsigned int)time(NULL);
    for(int i=0;i<workers;i++){
        WorkerCfg C;
        C.seed = base_seed + 1337u*i;
        // lightly jitter lr, temps, eps schedule across workers
        float lr = LR_DEFAULT * (0.7f + 0.6f*((float)rand()/RAND_MAX)); // ~[0.49..1.12]x
        float t  = 0.6f + 0.9f*((float)rand()/RAND_MAX);                // temp in [0.6..1.5]
        float e0 = EPS_START_DEFAULT, e1 = EPS_END_DEFAULT;
        C.eps_start = e0; C.eps_end = e1;
        C.boltz_temp = t; C.lr = lr; C.episodes = TRAIN_EPISODES; C.use_gpu = use_gpu;
        C.net = 0; C.eval_avg_reward = -1e9f; C.eval_successes = 0;
        ta[i].cfg = C;
        pthread_create(&th[i], NULL, worker_main, &ta[i]);
    }

    for(int i=0;i<workers;i++){ pthread_join(th[i], NULL); }

    // Pick best by avg reward, then successes
    int best=-1; float bestR=-1e9f; int bestS=-1;
    for(int i=0;i<workers;i++){
        if(ta[i].cfg.net==0) continue;
        if(ta[i].cfg.eval_avg_reward > bestR ||
          (ta[i].cfg.eval_avg_reward==bestR && ta[i].cfg.eval_successes>bestS)){
            best=i; bestR = ta[i].cfg.eval_avg_reward; bestS = ta[i].cfg.eval_successes;
        }
    }

    if(best<0){
        endwin(); fprintf(stderr,"All workers failed to produce a model.\n");
        free(th); free(ta); return 2;
    }

    // Optional save
    if(save_path){
        int ok = save_net(ta[best].cfg.net, save_path);
        attron(ok?COLOR_PAIR(C_GOAL):COLOR_PAIR(C_ERR));
        mvprintw(1,0, ok? "Saved best model to %s                      "
                         : "Save failed: %s                             ", save_path);
        attroff(ok?COLOR_PAIR(C_GOAL):COLOR_PAIR(C_ERR));
        refresh();
    }

    // Demo best
    attron(COLOR_PAIR(C_BOLD));
    mvprintw(2,0,"Best worker: %d | Eval Avg: %.3f | Success: %d/%d        ",
             best, ta[best].cfg.eval_avg_reward, ta[best].cfg.eval_successes, DEMO_EPS);
    attroff(COLOR_PAIR(C_BOLD));
    refresh();

    long long net = ta[best].cfg.net;

    float total=0.f; int steps_sum=0, succ=0;
    for(int ep=0; ep<DEMO_EPS; ep++){
        int steps; char path[64]={0};
        float r = run_episode(net, 0, &steps, path, 64, 0,1.0f, 0.0f, use_gpu, START_Y, START_X, 1);
        total+=r; steps_sum+=steps; if(r>=0.8f) succ++;
        attron(COLOR_PAIR(C_INFO));
        mvprintw(ROWS+4+ep,0,"Ep %d: R %6.2f | Steps %3d | Path: %-20s", ep, r, steps, path);
        attroff(COLOR_PAIR(C_INFO)); refresh();
    }
    float avg=total/DEMO_EPS;
    attron(COLOR_PAIR(C_BOLD));
    mvprintw(ROWS+4+DEMO_EPS,0,"Avg: %6.2f | Steps/Ep: %3d | Success: %d/%d",
             avg, steps_sum/DEMO_EPS, succ, DEMO_EPS);
    attroff(COLOR_PAIR(C_BOLD));

    getch();
    endwin();

    // Free all nets
    for(int i=0;i<workers;i++){ if(ta[i].cfg.net) Paracast_Free(ta[i].cfg.net); }
    free(th); free(ta);
    return 0;
}
