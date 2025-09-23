// grid_rl.c – Pretty Self-Contained C RL Grid Nav: Train, Save, Load, Square Grid Viz
// Compile: gcc grid_rl.c -lncurses -lm -o grid_rl
// Run: ./grid_rl train (trains/saves) | ./grid_rl load grid_policy.json (loads/runs) | ./grid_rl (default: train + demo)
#include <ncurses.h>  // Terminal UI
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   // usleep
#include <math.h>     // fmax
#include <time.h>     // srand/time
#include <getopt.h>   // CLI args

// Grid & Params
#define ROWS 5
#define COLS 5
#define STATES (ROWS * COLS)
#define ACTIONS 4
#define MAX_EPS_DEFAULT 1000
#define ALPHA_DEFAULT 0.1
#define GAMMA_DEFAULT 0.9
#define EPS_START_DEFAULT 1.0
#define EPS_END_DEFAULT 0.1
#define EPS_DECAY_DEFAULT 0.995
#define MAX_STEPS 100
#define NUM_DEMO_EPS 5
#define CELL_WIDTH 3  // For square look: +---+ per cell

int grid[ROWS][COLS] = {
    {0,0,0,1,0},
    {0,1,0,1,0},
    {0,0,0,0,2}, // Goal [2][4]
    {0,1,0,1,0},
    {0,0,0,0,0}  // Start [4][0]
};
int action_deltas[ACTIONS][2] = {{-1,0},{1,0},{0,-1},{0,1}}; // dy dx: U D L R
char action_chars[ACTIONS] = {'U', 'D', 'L', 'R'};

// Q-Table: states x actions
float q_table[STATES][ACTIONS];

// Color pairs: 1=wall(red), 2=goal(green), 3=agent(blue), 4=highlight(yellow), 5=bold white
#define COLOR_WALL 1
#define COLOR_GOAL 2
#define COLOR_AGENT 3
#define COLOR_HIGHLIGHT 4
#define COLOR_BOLD 5

// Helpers
int get_state(int y, int x) { return y * COLS + x; }
int is_valid(int y, int x) { return x >= 0 && x < COLS && y >= 0 && y < ROWS && grid[y][x] != 1; }
int is_goal(int y, int x) { return grid[y][x] == 2; }
int step(int y, int x, int a, int *ny, int *nx) {
    int dy = action_deltas[a][0], dx = action_deltas[a][1];
    *ny = y + dy; *nx = x + dx;
    if (!is_valid(*ny, *nx)) {
        *ny = y; *nx = x;
        return -1; // Bump
    }
    if (is_goal(*ny, *nx)) return 1; // Goal
    return -0.01; // Live
}
float max_q(int s) {
    float m = q_table[s][0];
    for (int a = 1; a < ACTIONS; a++) if (q_table[s][a] > m) m = q_table[s][a];
    return m;
}
int argmax_q(int s) {
    int best = 0;
    for (int a = 1; a < ACTIONS; a++) if (q_table[s][a] > q_table[s][best]) best = a;
    return best;
}

// Train Q-table
void train(int max_eps, float alpha, float gamma, float eps_start, float eps_end, float eps_decay) {
    srand(time(NULL));
    float eps = eps_start;
    int start_y = 4, start_x = 0;
    float total_rewards[MAX_EPS_DEFAULT];
    int ep;

    attron(COLOR_PAIR(COLOR_BOLD));
    mvprintw(0, 0, "🚀 Training Q-Table (%d eps, alpha=%.2f, gamma=%.2f)...", max_eps, alpha, gamma);
    attroff(COLOR_PAIR(COLOR_BOLD));
    refresh();

    for (ep = 0; ep < max_eps; ep++) {
        int y = start_y, x = start_x;
        float total_r = 0.0;
        int done = 0, steps = 0;

        while (!done && steps < MAX_STEPS) {
            int s = get_state(y, x);
            int a;
            if ((float)rand() / RAND_MAX < eps) {
                a = rand() % ACTIONS; // Explore
            } else {
                a = argmax_q(s); // Exploit
            }
            int ny, nx, r = step(y, x, a, &ny, &nx);
            int ns = get_state(ny, nx);
            q_table[s][a] += alpha * (r + gamma * max_q(ns) - q_table[s][a]);
            y = ny; x = nx;
            total_r += r;
            done = (r == 1);
            steps++;
        }
        total_rewards[ep] = total_r;
        eps = fmax(eps_end, eps * eps_decay);
        if (ep % 200 == 0) {
            attron(COLOR_PAIR(COLOR_HIGHLIGHT));
            mvprintw(1, 0, "Ep %d: R %.2f | Epsilon %.2f", ep, total_r, eps);
            attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
            refresh();
        }
    }
    float avg_last = 0;
    for (int i = max_eps - 100; i < max_eps; i++) avg_last += total_rewards[i];
    avg_last /= 100;
    attron(COLOR_PAIR(COLOR_BOLD));
    mvprintw(1, 0, "✅ Training done. Avg last 100: %.2f", avg_last);
    attroff(COLOR_PAIR(COLOR_BOLD));
    refresh();
    usleep(1000000); // Pause
}

// Save Q-table to JSON
void save_q(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        attron(COLOR_PAIR(COLOR_WALL));
        printw("❌ Save fail: %s\n", filename);
        attroff(COLOR_PAIR(COLOR_WALL));
        refresh();
        return;
    }
    fprintf(f, "{\n  \"q_table\": [");
    for (int s = 0; s < STATES; s++) {
        for (int a = 0; a < ACTIONS; a++) {
            fprintf(f, "%.6f", q_table[s][a]);
            if (s < STATES-1 || a < ACTIONS-1) fprintf(f, ", ");
        }
    }
    fprintf(f, "],\n  \"states\": %d,\n  \"actions\": %d\n}\n", STATES, ACTIONS);
    fclose(f);
    attron(COLOR_PAIR(COLOR_GOAL));
    printw("💾 Saved to %s\n", filename);
    attroff(COLOR_PAIR(COLOR_GOAL));
    refresh();
}

// Load Q-table from JSON (simple parse)
int load_q(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        attron(COLOR_PAIR(COLOR_WALL));
        printw("❌ Load fail: %s not found\n", filename);
        attroff(COLOR_PAIR(COLOR_WALL));
        refresh();
        return 0;
    }
    char buf[1024 * 10];
    fread(buf, 1, sizeof(buf), f);
    fclose(f);

    char *start = strstr(buf, "\"q_table\": [");
    if (!start) {
        attron(COLOR_PAIR(COLOR_WALL));
        printw("❌ Invalid JSON: No q_table\n");
        attroff(COLOR_PAIR(COLOR_WALL));
        refresh();
        return 0;
    }
    start += strlen("\"q_table\": [");
    int idx = 0;
    char *tok = strtok(start, "[ ,]");
    while (tok && idx < STATES * ACTIONS) {
        q_table[idx / ACTIONS][idx % ACTIONS] = atof(tok);
        tok = strtok(NULL, "[ ,]");
        idx++;
    }
    if (idx != STATES * ACTIONS) {
        attron(COLOR_PAIR(COLOR_WALL));
        printw("⚠️ Partial load: %d/%d values\n", idx, STATES * ACTIONS);
        attroff(COLOR_PAIR(COLOR_WALL));
        refresh();
        return 0;
    }
    attron(COLOR_PAIR(COLOR_GOAL));
    printw("✅ Loaded %dx%d Q-table from %s\n", STATES, ACTIONS, filename);
    attroff(COLOR_PAIR(COLOR_GOAL));
    refresh();
    return 1;
}

// Pretty draw: Square cells with borders + Q bar (ASCII-safe '=')
void draw_grid(int agent_x, int agent_y, float *q_vals) {
    clear();
    int offset = 0; // Left border

    // Top border
    attron(COLOR_PAIR(COLOR_HIGHLIGHT));
    mvprintw(0, offset, "+");
    for (int j = 0; j < COLS; j++) for (int k = 0; k < CELL_WIDTH; k++) printw("-");
    printw("+");
    attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
    printw("  Grid Nav RL");

    // Rows with cell borders
    for (int i = 0; i < ROWS; i++) {
        // Left border + row content
        attron(COLOR_PAIR(COLOR_HIGHLIGHT));
        mvprintw(i+1, offset, "|");
        attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
        for (int j = 0; j < COLS; j++) {
            attron(COLOR_PAIR(COLOR_HIGHLIGHT));
            printw("+");
            attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
            for (int k = 0; k < CELL_WIDTH - 1; k++) printw(" ");
            // Cell content (center)
            int cell_pos = offset + 1 + j * CELL_WIDTH + (CELL_WIDTH - 1) / 2;
            if (i==agent_y && j==agent_x) {
                attron(COLOR_PAIR(COLOR_AGENT));
                mvprintw(i+1, cell_pos, "@");
                attroff(COLOR_PAIR(COLOR_AGENT));
            } else if (grid[i][j]==1) {
                attron(COLOR_PAIR(COLOR_WALL));
                mvprintw(i+1, cell_pos, "#");
                attroff(COLOR_PAIR(COLOR_WALL));
            } else if (grid[i][j]==2) {
                attron(COLOR_PAIR(COLOR_GOAL));
                mvprintw(i+1, cell_pos, "G");
                attroff(COLOR_PAIR(COLOR_GOAL));
            } else {
                mvprintw(i+1, cell_pos, ".");
            }
            attron(COLOR_PAIR(COLOR_HIGHLIGHT));
            printw("+");
            attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
        }
        // Right border
        attron(COLOR_PAIR(COLOR_HIGHLIGHT));
        printw("|");
        attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
        printw("\n");
    }

    // Bottom border
    attron(COLOR_PAIR(COLOR_HIGHLIGHT));
    mvprintw(ROWS+1, offset, "+");
    for (int j = 0; j < COLS; j++) for (int k = 0; k < CELL_WIDTH; k++) printw("-");
    printw("+");
    attroff(COLOR_PAIR(COLOR_HIGHLIGHT));

    // Q bar (ASCII '=' for safe, scaled)
    if (q_vals) {
        mvprintw(ROWS+3, offset, "Q (U D L R): ");
        float max_qv = 0;
        for (int a=0; a<ACTIONS; a++) if (q_vals[a] > max_qv) max_qv = q_vals[a];
        for (int a=0; a<ACTIONS; a++) {
            int bar_len = (int)((q_vals[a] / max_qv) * 8); // 8-char max
            attron(COLOR_PAIR(COLOR_HIGHLIGHT));
            for (int b=0; b<bar_len; b++) printw("=");
            attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
            for (int b=bar_len; b<8; b++) printw("-");
            printw(" %.2f", q_vals[a]);
        }
        printw(" | Best: %c", action_chars[argmax_q(get_state(agent_y, agent_x))]);
    }
    refresh();
}

// Run episode with Q-table (greedy/random)
float run_episode(int trained, int *path_len, char *path_log, int max_log) {
    srand(time(NULL));
    int x=0, y=4; // Start
    int steps=0, max_steps=50;
    float total_r=0;
    char path[51] = {0};
    int log_idx = 0;
    float q[ACTIONS];
    draw_grid(x, y, NULL);
    usleep(200000);

    while (steps < max_steps) {
        int s = get_state(y, x);
        int a;
        if (!trained) {
            a = rand() % ACTIONS; // Random
        } else {
            a = argmax_q(s);
            for (int i=0; i<ACTIONS; i++) q[i] = q_table[s][i];
        }
        if (log_idx < max_log) {
            path[log_idx++] = action_chars[a];
            path[log_idx] = 0;
        }
        int ny, nx, r = step(y, x, a, &ny, &nx);
        if (r == -1) { total_r -=1; break; }
        x = nx; y = ny;
        steps++;
        total_r += r;
        if (r == 1) break;

        draw_grid(x, y, trained ? q : NULL);
        usleep(200000);
    }
    *path_len = steps;
    strcpy(path_log, path);
    return total_r;
}

int main(int argc, char **argv) {
    int opt, train_mode = 1, load_file = 0;
    int max_eps = MAX_EPS_DEFAULT;
    float alpha = ALPHA_DEFAULT, gamma = GAMMA_DEFAULT;
    float eps_start = EPS_START_DEFAULT, eps_end = EPS_END_DEFAULT, eps_decay = EPS_DECAY_DEFAULT;
    const char *filename = "grid_policy.json";

    // Parse args
    while ((opt = getopt(argc, argv, "t:e:a:g:s:d:f:")) != -1) {
        switch (opt) {
            case 't': train_mode = atoi(optarg); break;
            case 'e': max_eps = atoi(optarg); break;
            case 'a': alpha = atof(optarg); break;
            case 'g': gamma = atof(optarg); break;
            case 's': eps_start = atof(optarg); break;
            case 'd': eps_decay = atof(optarg); break;
            case 'f': filename = optarg; load_file = 1; break;
            default: printw("Usage: ./grid_rl [-t 1] [-e 1000] [-a 0.1] [-g 0.9] [-s 1.0] [-d 0.995] [-f file.json]\n"); return 1;
        }
    }

    initscr();
    noecho();
    curs_set(0);
    start_color();
    init_pair(COLOR_WALL, COLOR_RED, COLOR_BLACK);      // Red walls
    init_pair(COLOR_GOAL, COLOR_GREEN, COLOR_BLACK);    // Green goal
    init_pair(COLOR_AGENT, COLOR_BLUE, COLOR_BLACK);    // Blue agent
    init_pair(COLOR_HIGHLIGHT, COLOR_YELLOW, COLOR_BLACK); // Yellow accents
    init_pair(COLOR_BOLD, COLOR_WHITE, COLOR_BLACK);    // Bold white

    int trained = 0;
    if (load_file) {
        trained = load_q(filename);
        train_mode = 0;
    }
    if (train_mode) {
        train(max_eps, alpha, gamma, eps_start, eps_end, eps_decay);
        save_q(filename);
        trained = 1;
    }

    // Pretty demo header
    attron(COLOR_PAIR(COLOR_BOLD));
    mvprintw(ROWS+4, 0, "🎮 RL Demo: %d Episodes | Trained: %s", NUM_DEMO_EPS, trained ? "Yes" : "Random");
    attroff(COLOR_PAIR(COLOR_BOLD));

    // Demo runs
    float avg_r = 0;
    int total_steps = 0;
    for (int ep = 0; ep < NUM_DEMO_EPS; ep++) {
        int steps;
        char path_log[51];
        float r = run_episode(trained, &steps, path_log, 50);
        avg_r += r;
        total_steps += steps;
        attron(COLOR_PAIR(COLOR_HIGHLIGHT));
        mvprintw(ROWS+5 + ep, 0, "Ep %d: R %.1f | Steps %d | Path: %s", ep, r, steps, path_log);
        attroff(COLOR_PAIR(COLOR_HIGHLIGHT));
        refresh();
        usleep(500000);
    }
    avg_r /= NUM_DEMO_EPS;
    attron(COLOR_PAIR(COLOR_BOLD));
    mvprintw(ROWS+5 + NUM_DEMO_EPS, 0, "📊 Avg: %.1f Reward | %d Steps/Ep", avg_r, total_steps / NUM_DEMO_EPS);
    attroff(COLOR_PAIR(COLOR_BOLD));

    getch();
    endwin();
    return 0;
}