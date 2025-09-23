// pendulum_tabular_paragon.c
// Discretized inverted pendulum: fast tabular Q-learning + Paragon NN fit (stabilized)
//
// Build (from dist/):
//   gcc pendulum_tabular_paragon.c -I. -L. -lparacast -lm -o pendulum_tabular_paragon \
//       -Wl,-rpath,'$ORIGIN'
//
// Run:
//   ./pendulum_tabular_paragon [--episodes N] [--bins B] [--gpu] [--save qtable.csv] [--savefit fit.csv]
//
// What it does:
// 1) Trains tabular Q on a discretized inverted pendulum with 3 actions ([-τ,0,+τ]).
// 2) Greedy rollout from upright; prints return and saves rollout to CSV.
// 3) Fits a small Paragon MLP (supervised) to the learned Q(s,·) with standardized targets,
//    reporting MSE back in original Q units.
//
// Changes vs previous:
// - Dynamics: dt=0.02, torque=3.0, clamp dtheta, smoother cosine reward
// - Q-learn: lr=0.20 default, more episodes recommended
// - Paragon fit: linear head, target standardization (mean/std) -> proper regression

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <time.h>

#include "libparacast.h"  // Paragon C-ABI

// ---------- Config ----------
typedef struct {
    float dt;        // integration step
    float g;         // gravity
    float l;         // length
    float m;         // mass
    float torque;    // |u| magnitude
    int   max_steps; // episode length
    float theta_clip;// clamp for binning
} PendCfg;

typedef struct {
    // discretization
    int bins_th;
    int bins_dth;
    // Q-table
    int n_states;
    int n_actions;
    float* Q; // [n_states * 3]
    // env state
    float theta;
    float dtheta;
    int steps;
    PendCfg cfg;
} PendEnv;

static inline float frand(){ return (float)rand()/(float)RAND_MAX; }
static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
static inline float wrap_pi(float x){ // wrap to [-pi, pi)
    float y = fmodf(x + (float)M_PI, 2.0f*(float)M_PI);
    if(y < 0) y += 2.0f*(float)M_PI;
    return y - (float)M_PI;
}

// ---------- Env ----------
static void pend_reset(PendEnv* E){
    E->theta = (frand()-0.5f)*0.2f;
    E->dtheta = (frand()-0.5f)*0.2f;
    E->steps = 0;
}
static void pend_init(PendEnv* E, PendCfg cfg, int bins){
    E->cfg = cfg;
    E->bins_th = bins; E->bins_dth = bins;
    E->n_actions = 3;
    E->n_states = E->bins_th * E->bins_dth;
    E->Q = (float*)calloc((size_t)E->n_states*(size_t)E->n_actions, sizeof(float));
    pend_reset(E);
}
static void pend_free(PendEnv* E){ free(E->Q); E->Q=NULL; }

// index mapping
static inline int obs_index(PendEnv* E){
    float th = clampf(E->theta, -E->cfg.theta_clip, E->cfg.theta_clip);
    float dth = clampf(E->dtheta, -8.0f, 8.0f);
    int th_bin  = (int)((th  + E->cfg.theta_clip)/(2.0f*E->cfg.theta_clip) * (E->bins_th -1));
    int dth_bin = (int)((dth + 8.0f)               /(16.0f)                 * (E->bins_dth-1));
    if(th_bin  < 0) th_bin=0; if(th_bin>=E->bins_th) th_bin=E->bins_th-1;
    if(dth_bin < 0) dth_bin=0; if(dth_bin>=E->bins_dth) dth_bin=E->bins_dth-1;
    return th_bin * E->bins_dth + dth_bin;
}
static inline float* Qs(PendEnv* E, int s){ return &E->Q[s*E->n_actions]; }

// dynamics step
static void pend_step_dynamics(PendEnv* E, float u){
    // theta_ddot = (g/l) sin(theta) + u/(m l^2)
    float thdd = (E->cfg.g/E->cfg.l)*sinf(E->theta) + u/(E->cfg.m*E->cfg.l*E->cfg.l);
    E->dtheta += E->cfg.dt * thdd;
    E->dtheta = clampf(E->dtheta, -12.0f, 12.0f);   // clamp angular velocity
    E->theta  += E->cfg.dt * E->dtheta;
    E->theta   = wrap_pi(E->theta);
}
static void pend_step(PendEnv* E, int a, int* s_next, float* r, int* done){
    E->steps++;
    float u = (a==0? -E->cfg.torque : (a==1? 0.0f : E->cfg.torque));
    pend_step_dynamics(E, u);
    // Smooth "upright + low speed" reward
    float upright  = 0.5f*(1.0f + cosf(E->theta));  // 1 at 0 rad, 0 at pi
    float speedpen = 0.02f*(E->dtheta*E->dtheta);
    *r = upright - speedpen;                        // typically in [-1, 1]
    *done = (E->steps >= E->cfg.max_steps);
    *s_next = obs_index(E);
}

// ---------- Q-learning ----------
static int   argmax3(float* q){ int b=0; if(q[1]>q[b]) b=1; if(q[2]>q[b]) b=2; return b; }
static float max3   (float* q){ float m=q[0]; if(q[1]>m)m=q[1]; if(q[2]>m)m=q[2]; return m; }

typedef struct {
    int episodes;
    float gamma;
    float lr;
    float eps_start, eps_end;
} TrainCfg;

static float anneal_eps(int ep, int episodes, float e0, float e1){
    float t = (episodes>1)? (float)ep/(float)(episodes-1) : 1.0f;
    return e0*(1.0f-t) + e1*t;
}

static void train_q(PendEnv* E, TrainCfg T, float* returns){
    for(int ep=0; ep<T.episodes; ep++){
        pend_reset(E);
        int s = obs_index(E);
        int done=0; float G=0.0f;
        float eps = anneal_eps(ep, T.episodes, T.eps_start, T.eps_end);
        while(!done){
            float* q = Qs(E,s);
            int a = (frand()<eps)? (rand()%E->n_actions) : argmax3(q);
            int s2; float r; int d;
            pend_step(E, a, &s2, &r, &d);
            float td_target = r + (d? 0.0f : T.gamma * max3(Qs(E,s2)));
            q[a] = (1.0f - T.lr)*q[a] + T.lr*td_target;  // Polyak update
            s=s2; G+=r; done=d;
        }
        returns[ep]=G;
        if((ep+1)%100==0){
            printf("[train] ep %4d/%4d  eps=%.3f  G=%.3f\n", ep+1, T.episodes, eps, G);
        }
    }
}

// greedy rollout for inspection (from a fixed start near upright)
static float rollout_greedy(PendEnv* E, int max_steps, float start_theta, float start_dtheta,
                            float* theta_hist, float* dtheta_hist, int* out_steps)
{
    E->theta = start_theta; E->dtheta = start_dtheta; E->steps=0;
    int s = obs_index(E);
    float G=0.0f; int steps=0;
    for(int t=0;t<max_steps;t++){
        float* q = Qs(E,s);
        int a = argmax3(q);
        int s2; float r; int done;
        pend_step(E, a, &s2, &r, &done);
        if(theta_hist)  theta_hist[t]=E->theta;
        if(dtheta_hist) dtheta_hist[t]=E->dtheta;
        G+=r; steps++; s=s2;
        if(done) break;
    }
    if(out_steps) *out_steps = steps;
    return G;
}

// ---------- Paragon fit: supervised regression Q(s)->q_vec ----------
static long long make_paragon_net(int use_gpu, unsigned int seed){
    // Input: 2 features (normalized), Hidden: 32 tanh, Output: 3 linear (no tanh cap)
    int layers[] = {2,1, 32,1, 3,1};
    const char* acts[] = {"linear","tanh","linear"};   // linear head
    unsigned char fully[] = {1,1,1};
    long long net = Paracast_NewNetwork(layers,3,(char**)acts,3,fully,3,seed);
    if(!net) return 0;
    if(use_gpu){
        if(!Paracast_InitGPU(net)){
            const char* err = Paracast_GetLastError();
            fprintf(stderr,"GPU init failed (CPU fallback): %s\n", err?err:"(nil)");
        }
    }
    return net;
}

static void state_to_xy(int s, int bins, float* x2){ // normalized features
    int th_bin = s / bins;
    int d_bin  = s % bins;
    float th = - (float)M_PI + (2.0f*(float)M_PI) * ((float)th_bin/(float)(bins-1));
    float dth = -8.0f + 16.0f * ((float)d_bin/(float)(bins-1));
    // normalize to [-1,1] roughly
    x2[0] = th / (float)M_PI;
    x2[1] = dth / 8.0f;
}

// Fit NN to Q-table samples (robust version).
// - Fixed target scaling (Q_SCALE) to avoid zero-variance issues
// - Small LR + conservative grad clip
// - NaN/Inf guards on inputs/outputs
// - Caps sample count for stability
static float fit_paragon_to_qtable(PendEnv* E, int samples, int epochs, float lr, int use_gpu){
    long long net = make_paragon_net(use_gpu, (unsigned)time(NULL));
    if(!net){ fprintf(stderr,"Paragon NewNetwork failed\n"); return -1.0f; }

    // Keep targets in a tame range for a linear head.
    // With your reward/horizon, Q is roughly [0..~200]; this scale works well.
    const float Q_SCALE = 200.0f;

    // Safety: don’t train on a gigantic “samples”; cap to something sane per epoch.
    if (samples > 4096) samples = 4096;

    // Use a small LR; if you still see NaNs, try 0.001 or 0.0005
    if (lr <= 0.f || lr > 0.02f) lr = 0.0025f;

    for(int ep=1; ep<=epochs; ep++){
        double mse_orig = 0.0;
        for(int k=0;k<samples;k++){
            // Random state index
            int s = rand()%E->n_states;

            // Input features in ~[-1,1]
            float x[2]; state_to_xy(s, E->bins_dth, x);
            if (!isfinite(x[0]) || !isfinite(x[1])) continue;

            // Targets scaled to ~[-1,1]
            float y[3];
            for(int i=0;i<3;i++){
                float q = E->Q[s*3+i];
                // Guard: replace any NaN/Inf in the table with 0
                if (!isfinite(q)) q = 0.f;
                q /= Q_SCALE;
                if (q >  1.f) q =  1.f;
                if (q < -1.f) q = -1.f;
                y[i] = q;
            }

            // Train (conservative clip)
            int ok = Paracast_Train(
                net,
                x, 2, 1,
                y, 3, 1,
                1,               // batch
                1,               // epochs per call
                lr,
                use_gpu ? 1 : 0, // device
                -0.25f, 0.25f    // gradient clip low, high
            );
            if(!ok){
                const char* err=Paracast_GetLastError();
                fprintf(stderr,"Train err: %s\n", err?err:"(nil)");
                // Try to continue; if it keeps happening, rerun with --gpu off
                continue;
            }

            // Accumulate MSE in original Q units
            if(Paracast_Forward(net, x,2,1)){
                float out[3]; Paracast_GetOutput(net, out, 3);
                int bad=0; for(int i=0;i<3;i++) if(!isfinite(out[i])) { bad=1; break; }
                if(!bad){
                    for(int i=0;i<3;i++){
                        float e = (y[i] - out[i]) * Q_SCALE; // back to Q-units
                        mse_orig += 0.5 * (double)e * (double)e;
                    }
                }
            }
        }

        mse_orig /= (double)(samples*3);
        if(ep%50==0){
            // NOTE: this is MSE in original Q units (so 0–200 scale makes sense)
            printf("[fit] epoch %4d  MSE≈%.6f (Q units)\n", ep, (float)mse_orig);
        }
    }

    // Final evaluation on 1024 random states
    double mse_eval = 0.0;
    int evalN = 1024;
    for(int k=0;k<evalN;k++){
        int s = rand()%E->n_states;
        float x[2]; state_to_xy(s, E->bins_dth, x);
        if (!isfinite(x[0]) || !isfinite(x[1])) continue;

        float y[3];
        for(int i=0;i<3;i++){
            float q = E->Q[s*3+i]; if(!isfinite(q)) q = 0.f;
            y[i] = q / Q_SCALE;
            if (y[i] >  1.f) y[i] =  1.f;
            if (y[i] < -1.f) y[i] = -1.f;
        }

        if(Paracast_Forward(net, x,2,1)){
            float out[3]; Paracast_GetOutput(net, out, 3);
            int bad=0; for(int i=0;i<3;i++) if(!isfinite(out[i])) { bad=1; break; }
            if(!bad){
                for(int i=0;i<3;i++){
                    float e = (y[i]-out[i]) * Q_SCALE;
                    mse_eval += 0.5 * (double)e * (double)e;
                }
            }
        }
    }
    mse_eval /= (double)(evalN*3);
    printf("[fit] eval MSE≈%.6f (original Q units) on %d random states\n", (float)mse_eval, evalN);

    Paracast_Free(net);
    return (float)mse_eval;
}



// ---------- CSV helpers ----------
static void save_qtable_csv(const char* path, PendEnv* E){
    FILE* f=fopen(path,"w");
    if(!f){ perror("open qtable.csv"); return; }
    fprintf(f,"state,th_bin,dth_bin,Q_neg,Q_zero,Q_pos\n");
    for(int s=0;s<E->n_states;s++){
        int th_bin = s / E->bins_dth;
        int d_bin  = s % E->bins_dth;
        float* q = &E->Q[s*3];
        fprintf(f,"%d,%d,%d,%.6f,%.6f,%.6f\n", s,th_bin,d_bin,q[0],q[1],q[2]);
    }
    fclose(f);
    printf("Saved Q-table to %s\n", path);
}
static void save_rollout_csv(const char* path, const float* th, const float* dth, int T){
    FILE* f=fopen(path,"w");
    if(!f){ perror("open rollout.csv"); return; }
    fprintf(f,"step,theta,dtheta\n");
    for(int t=0;t<T;t++) fprintf(f,"%d,%.6f,%.6f\n", t, th[t], dth[t]);
    fclose(f);
    printf("Saved rollout to %s\n", path);
}
static void save_fit_metrics_csv(const char* path, float mse){
    FILE* f=fopen(path,"w");
    if(!f){ perror("open fit.csv"); return; }
    fprintf(f,"mse\n%.8f\n", mse);
    fclose(f);
    printf("Saved fit metrics to %s\n", path);
}

// ---------- Main ----------
int main(int argc, char** argv){
    // Defaults
    int episodes = 1500;           // feel free to pass --episodes 3000
    int bins = 19;
    int use_gpu = 0;
    const char* save_q = NULL;
    const char* save_fit = NULL;

    static struct option opts[] = {
        {"episodes", required_argument, 0, 'e'},
        {"bins",     required_argument, 0, 'b'},
        {"gpu",      no_argument,       0, 'g'},
        {"save",     required_argument, 0, 's'},
        {"savefit",  required_argument, 0, 'f'},
        {0,0,0,0}
    };
    int c;
    while((c=getopt_long(argc, argv, "e:b:gs:f:", opts, NULL))!=-1){
        if(c=='e') episodes = atoi(optarg);
        else if(c=='b') bins = atoi(optarg);
        else if(c=='g') use_gpu = 1;
        else if(c=='s') save_q = optarg;
        else if(c=='f') save_fit = optarg;
    }

    srand((unsigned)time(NULL));

    // Env config (tighter dt, stronger torque)
    PendCfg cfg = {
        .dt = 0.02f, .g = 9.81f, .l = 1.0f, .m = 1.0f,
        .torque = 3.0f, .max_steps = 200, .theta_clip = (float)M_PI
    };
    PendEnv env; pend_init(&env, cfg, bins);

    // Q-learning config (tuned for stability)
    TrainCfg T = { .episodes = episodes, .gamma = 0.99f, .lr = 0.20f,
                   .eps_start=1.0f, .eps_end=0.05f };

    float* returns = (float*)malloc(sizeof(float)*episodes);
    printf("Training tabular Q: episodes=%d bins=%d use_gpu_fit=%s\n", episodes, bins, use_gpu?"yes":"no");
    train_q(&env, T, returns);

    // Greedy rollout from upright (0,0)
    float* TH = (float*)malloc(sizeof(float)*cfg.max_steps);
    float* DTH= (float*)malloc(sizeof(float)*cfg.max_steps);
    int Tsteps=0;
    float G = rollout_greedy(&env, cfg.max_steps, 0.0f, 0.0f, TH, DTH, &Tsteps);
    printf("Greedy rollout: steps=%d  return=%.3f\n", Tsteps, G);

    // Optional saves
    if(save_q){ save_qtable_csv(save_q, &env); }
    save_rollout_csv("rollout.csv", TH, DTH, Tsteps);

    // Paragon fit: take the learned table and fit a small MLP (supervised)
    printf("Fitting Paragon NN to Q-table (supervised)…\n");
    float mse = fit_paragon_to_qtable(&env, /*samples*/env.n_states, /*epochs*/400, /*lr*/0.005f, use_gpu);
    if(save_fit){ save_fit_metrics_csv(save_fit, mse); }

    // Print last 50 returns average
    float avg_last=0.f; int cnt = (episodes<50)?episodes:50;
    for(int i=episodes-cnt;i<episodes;i++) avg_last += returns[i];
    avg_last/= (float)cnt;
    printf("Avg return over last %d episodes: %.3f\n", cnt, avg_last);

    // Cleanup
    free(returns); free(TH); free(DTH);
    pend_free(&env);
    return 0;
}
