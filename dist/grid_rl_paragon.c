// grid_rl_paragon_fixed.c – Fixed Grid RL with Paragon
// Build: gcc grid_rl_paragon_fixed.c -I. -L. -lparacast -lncursesw -lm -Wl,-rpath,'$ORIGIN' -o grid_rl_paragon_fixed
#include <locale.h>
#include <ncursesw/ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include "libparacast.h"
// ===== Grid Configuration =====
#define ROWS 5
#define COLS 5
#define ACTIONS 4
#define NUM_STATES (ROWS * COLS)
static int grid[ROWS][COLS] = {
    {0,0,0,1,0},
    {0,1,0,1,0},
    {0,0,0,0,2}, // Goal at [2][4]
    {0,1,0,1,0},
    {0,0,0,0,0} // Start at [4][0]
};
static int action_deltas[ACTIONS][2] = {{-1,0},{1,0},{0,-1},{0,1}}; // U D L R
static char action_chars[ACTIONS] = {'U','D','L','R'};
// ===== Colors =====
#define COLOR_WALL 1
#define COLOR_GOAL 2
#define COLOR_AGENT 3
#define COLOR_HILITE 4
#define COLOR_BOLD 5
#define COLOR_ERROR 6
// ===== FIXED Training Parameters =====
#define MAX_STEPS 50
#define DEMO_EPS 5
#define TRAINING_EPISODES 3000 // More episodes needed
#define EPS_START 0.9f // Start with high exploration
#define EPS_END 0.01f // Low final exploration
#define EPS_DECAY 0.9995f // Slower decay
#define GAMMA 0.99f // Higher discount factor
#define LEARNING_RATE 0.0005f // Much lower learning rate
#define BATCH_SIZE 1
#define EPOCHS_PER_UPDATE 5
// ===== FIXED Reward Structure =====
#define R_GOAL 100.0f // Much higher goal reward
#define R_BUMP -10.0f // Higher wall penalty
#define R_STEP -1.0f // Step penalty to encourage efficiency
#define R_PROGRESS 5.0f // Reward for getting closer
// Global variables
static int START_Y = 4, START_X = 0;
static int goal_y = 2, goal_x = 4;
static int debug_mode = 0;
// ===== Helper Functions =====
static inline int in_bounds(int y, int x) {
return (x >= 0 && x < COLS && y >= 0 && y < ROWS && grid[y][x] != 1);
}
static inline int is_goal(int y, int x) {
return grid[y][x] == 2;
}
static inline int get_state(int y, int x) {
return y * COLS + x;
}
static inline int manhattan_distance(int y1, int x1, int y2, int x2) {
return abs(y1 - y2) + abs(x1 - x2);
}
// FIXED: Better reward function
static float step_env(int y, int x, int action, int *ny, int *nx) {
int old_dist = manhattan_distance(y, x, goal_y, goal_x);
int dy = action_deltas[action][0];
int dx = action_deltas[action][1];
*ny = y + dy;
*nx = x + dx;
// Hit wall or boundary
if (!in_bounds(*ny, *nx)) {
*ny = y; // Stay in place
*nx = x;
return R_BUMP;
    }
// Reached goal
if (is_goal(*ny, *nx)) {
return R_GOAL;
    }
// Normal move - reward getting closer to goal
int new_dist = manhattan_distance(*ny, *nx, goal_y, goal_x);
float reward = R_STEP; // Base step penalty
if (new_dist < old_dist) {
        reward += R_PROGRESS; // Reward progress toward goal
    }
return reward;
}
// FIXED: Proper state encoding
static void encode_state(int state, float *buffer) {
for (int i = 0; i < NUM_STATES; i++) {
buffer[i] = 0.0f;
    }
if (state >= 0 && state < NUM_STATES) {
buffer[state] = 1.0f;
    }
}
// FIXED: Better Q-network forward pass
static int get_q_values(long long net, int state, float *q_values) {
float state_input[NUM_STATES];
encode_state(state, state_input);
if (!Paracast_Forward(net, state_input, NUM_STATES, 1)) {
if (debug_mode) {
const char* err = Paracast_GetLastError();
mvprintw(20, 0, "Forward pass failed: %s", err ? err : "unknown");
refresh();
        }
return 0;
    }
int outputs = Paracast_GetOutput(net, q_values, ACTIONS);
if (outputs != ACTIONS) {
if (debug_mode) {
mvprintw(20, 0, "Expected %d outputs, got %d", ACTIONS, outputs);
refresh();
        }
return 0;
    }
return 1;
}
// Find best action
static int get_best_action(const float *q_values) {
int best = 0;
for (int i = 1; i < ACTIONS; i++) {
if (q_values[i] > q_values[best]) {
            best = i;
        }
    }
return best;
}
// FIXED: Much better visualization
static void draw_grid(int agent_x, int agent_y, const float *q_values,
int episode, float epsilon, float total_reward) {
clear();
// Title and info
attron(COLOR_PAIR(COLOR_BOLD));
mvprintw(0, 0, "=== GRID RL WITH PARAGON ===");
attroff(COLOR_PAIR(COLOR_BOLD));
if (episode >= 0) {
mvprintw(1, 0, "Episode: %d | Epsilon: %.3f | Total Reward: %.1f",
                episode, epsilon, total_reward);
    }
// Grid border
attron(COLOR_PAIR(COLOR_HILITE));
mvprintw(3, 0, "+-------+");
for (int i = 0; i < ROWS; i++) {
mvprintw(4 + i, 0, "|");
mvprintw(4 + i, 7, "|");
    }
mvprintw(4 + ROWS, 0, "+-------+");
attroff(COLOR_PAIR(COLOR_HILITE));
// Draw grid contents
for (int y = 0; y < ROWS; y++) {
for (int x = 0; x < COLS; x++) {
int screen_y = 4 + y;
int screen_x = 1 + x;
if (y == agent_y && x == agent_x) {
attron(COLOR_PAIR(COLOR_AGENT));
mvprintw(screen_y, screen_x, "@");
attroff(COLOR_PAIR(COLOR_AGENT));
            } else if (grid[y][x] == 1) {
attron(COLOR_PAIR(COLOR_WALL));
mvprintw(screen_y, screen_x, "#");
attroff(COLOR_PAIR(COLOR_WALL));
            } else if (grid[y][x] == 2) {
attron(COLOR_PAIR(COLOR_GOAL));
mvprintw(screen_y, screen_x, "G");
attroff(COLOR_PAIR(COLOR_GOAL));
            } else {
mvprintw(screen_y, screen_x, ".");
            }
        }
    }
// Q-values display
if (q_values) {
mvprintw(10, 0, "Q-Values:");
for (int i = 0; i < ACTIONS; i++) {
attron(i == get_best_action(q_values) ? COLOR_PAIR(COLOR_GOAL) : COLOR_PAIR(COLOR_HILITE));
mvprintw(11 + i, 0, "%c: %8.2f", action_chars[i], q_values[i]);
attroff(i == get_best_action(q_values) ? COLOR_PAIR(COLOR_GOAL) : COLOR_PAIR(COLOR_HILITE));
        }
    }
refresh();
}
// FIXED: Proper Q-learning update
static int train_network(long long net, int state, int action, float reward,
int next_state, int done, int use_gpu) {
float state_input[NUM_STATES], next_state_input[NUM_STATES];
float q_current[ACTIONS], q_next[ACTIONS];
// Get current Q-values
if (!get_q_values(net, state, q_current)) {
return 0;
    }
// Compute TD target
float td_target;
if (done) {
        td_target = reward; // Terminal state
    } else {
if (!get_q_values(net, next_state, q_next)) {
return 0;
        }
// Find max Q-value for next state
float max_q_next = q_next[0];
for (int i = 1; i < ACTIONS; i++) {
if (q_next[i] > max_q_next) {
                max_q_next = q_next[i];
            }
        }
        td_target = reward + GAMMA * max_q_next;
    }
// Create target vector - only update the action taken
float target[ACTIONS];
for (int i = 0; i < ACTIONS; i++) {
target[i] = q_current[i];
    }
target[action] = td_target;
// Train the network
encode_state(state, state_input);
int success = Paracast_Train(
        net,
        state_input, NUM_STATES, 1, // input
        target, ACTIONS, 1, // target
        BATCH_SIZE, // batch size
        EPOCHS_PER_UPDATE, // epochs
        LEARNING_RATE, // learning rate
        use_gpu ? 1 : 0, // use GPU
1.0f, -1.0f // gradient clipping FIXED: upper=1.0, lower=-1.0
    );
if (!success && debug_mode) {
const char* err = Paracast_GetLastError();
mvprintw(22, 0, "Training failed: %s", err ? err : "unknown");
refresh();
    }
return success;
}
// FIXED: Better episode runner
static float run_episode(long long net, int training_mode, int use_gpu,
int episode_num, int *steps_out, char *action_log, int log_size) {
int x = START_X, y = START_Y;
float total_reward = 0.0f;
int steps = 0;
// Epsilon decay
float epsilon = 0.0f;
if (training_mode) {
        epsilon = EPS_END + (EPS_START - EPS_END) * expf(-episode_num / 1000.0f);
        epsilon = fmaxf(epsilon, EPS_END);
    }
if (action_log) action_log[0] = '\0';
while (steps < MAX_STEPS) {
int current_state = get_state(y, x);
float q_values[ACTIONS];
int action;
// Choose action
if (training_mode && ((float)rand() / RAND_MAX) < epsilon) {
            action = rand() % ACTIONS; // Random exploration
        } else {
if (!get_q_values(net, current_state, q_values)) {
                action = rand() % ACTIONS; // Fallback
            } else {
                action = get_best_action(q_values);
            }
        }
// Log action
if (action_log && steps < log_size - 1) {
action_log[steps] = action_chars[action];
action_log[steps + 1] = '\0';
        }
// Take action
int next_y, next_x;
float reward = step_env(y, x, action, &next_y, &next_x);
int next_state = get_state(next_y, next_x);
        total_reward += reward;
// Check if episode is done
int done = (reward == R_GOAL) || (reward == R_BUMP);
// Train if in training mode
if (training_mode) {
if (!train_network(net, current_state, action, reward, next_state, done, use_gpu)) {
if (debug_mode) {
mvprintw(23, 0, "Training failed at step %d", steps);
refresh();
                }
            }
        }
// Update position
        x = next_x;
        y = next_y;
        steps++;
// Visualization (only sometimes during training)
if (!training_mode || (training_mode && episode_num % 100 == 0)) {
if (get_q_values(net, get_state(y, x), q_values)) {
draw_grid(x, y, q_values, training_mode ? episode_num : -1, epsilon, total_reward);
usleep(training_mode ? 50000 : 200000);
            }
        }
// Check terminal conditions
if (done) break;
    }
if (steps_out) *steps_out = steps;
return total_reward;
}
// FIXED: Better network architecture
static long long create_network(int use_gpu) {
// Better architecture: input -> hidden1 -> hidden2 -> output
int layer_config[] = {
        NUM_STATES, 1, // Input layer: 25 neurons
128, 1, // Hidden layer 1: 128 neurons
64, 1, // Hidden layer 2: 64 neurons
        ACTIONS, 1 // Output layer: 4 neurons
    };
const char* activations[] = {"linear", "relu", "relu", "linear"};
unsigned char fully_connected[] = {1, 1, 1, 1};
long long net = Paracast_NewNetwork(
        layer_config, 4,
        (char**)activations, 4,
        fully_connected, 4,
        use_gpu ? 1 : 0 // FIXED: enable WebGPU based on use_gpu
    );
if (net == 0) {
const char* err = Paracast_GetLastError();
fprintf(stderr, "Network creation failed: %s\n", err ? err : "unknown");
return 0;
    }
if (use_gpu) {
if (!Paracast_InitGPU(net)) {
const char* err = Paracast_GetLastError();
fprintf(stderr, "GPU init failed, using CPU: %s\n", err ? err : "unknown");
        }
    }
return net;
}
static long long load_network_from_file(const char* path, int use_gpu) {
    FILE* f = fopen(path, "rb");
if (!f) {
fprintf(stderr, "Cannot open %s\n", path);
return 0;
    }
fseek(f, 0, SEEK_END);
long len = ftell(f);
fseek(f, 0, SEEK_SET);
char* buf = malloc(len + 1);
if (!buf) {
fclose(f);
return 0;
    }
fread(buf, 1, len, f);
buf[len] = '\0';
fclose(f);
long long net = Paracast_LoadFromJSON(buf, use_gpu ? 1 : 0);
if (net == 0) {
const char* err = Paracast_GetLastError();
fprintf(stderr, "Load failed: %s\n", err ? err : "unknown");
    }
free(buf);
return net;
}
static void parse_start_position(const char* s) {
int y, x;
if (sscanf(s, "%d,%d", &y, &x) == 2) {
if (y >= 0 && y < ROWS && x >= 0 && x < COLS && grid[y][x] != 1) {
            START_Y = y;
            START_X = x;
        }
    }
}
int main(int argc, char** argv) {
srand((unsigned int)time(NULL));
// Parse arguments
int use_gpu = 0, demo_only = 0;
const char* save_path = NULL;
const char* load_path = NULL;
static struct option opts[] = {
        {"gpu", no_argument, 0, 'g'},
        {"demo", no_argument, 0, 'd'},
        {"save", required_argument, 0, 's'},
        {"load", required_argument, 0, 'l'},
        {"start", required_argument, 0, 'S'},
        {"debug", no_argument, 0, 'D'},
        {0, 0, 0, 0}
    };
int c;
while ((c = getopt_long(argc, argv, "gds:l:S:D", opts, NULL)) != -1) {
switch (c) {
case 'g': use_gpu = 1; break;
case 'd': demo_only = 1; break;
case 's': save_path = optarg; break;
case 'l': load_path = optarg; break;
case 'S': parse_start_position(optarg); break;
case 'D': debug_mode = 1; break;
        }
    }
// Initialize ncurses
setlocale(LC_ALL, "");
initscr();
noecho();
curs_set(0);
start_color();
init_pair(COLOR_WALL, COLOR_RED, COLOR_BLACK);
init_pair(COLOR_GOAL, COLOR_GREEN, COLOR_BLACK);
init_pair(COLOR_AGENT, COLOR_CYAN, COLOR_BLACK);
init_pair(COLOR_HILITE, COLOR_YELLOW, COLOR_BLACK);
init_pair(COLOR_BOLD, COLOR_WHITE, COLOR_BLACK);
init_pair(COLOR_ERROR, COLOR_MAGENTA, COLOR_BLACK);
// Create/load network
long long net = 0;
if (load_path) {
        net = load_network_from_file(load_path, use_gpu);
if (!net) {
endwin();
return 1;
        }
    } else {
        net = create_network(use_gpu);
if (!net) {
endwin();
return 1;
        }
    }
int trained = (load_path != NULL);
// TRAINING PHASE
if (!demo_only && !trained) {
attron(COLOR_PAIR(COLOR_BOLD));
mvprintw(0, 0, "TRAINING PARAGON Q-NETWORK...");
mvprintw(1, 0, "Episodes: %d | GPU: %s | Debug: %s",
                TRAINING_EPISODES, use_gpu ? "ON" : "OFF", debug_mode ? "ON" : "OFF");
attroff(COLOR_PAIR(COLOR_BOLD));
refresh();
float total_training_reward = 0.0f;
int successes = 0;
int total_steps_all = 0;
for (int episode = 0; episode < TRAINING_EPISODES; episode++) {
int steps;
char action_path[64];
float reward = run_episode(net, 1, use_gpu, episode, &steps, action_path, sizeof(action_path));
            total_training_reward += reward;
            total_steps_all += steps;
if (reward > 50.0f) successes++; // Consider success if reward > 50
// Progress update
if (episode % 100 == 0) {
float avg_reward = total_training_reward / (episode + 1);
float success_rate = (float)successes / (episode + 1) * 100.0f;
attron(COLOR_PAIR(COLOR_HILITE));
mvprintw(16, 0, "Episode %4d | Avg R: %.1f | Success: %.1f%% | Last R=%.1f Steps=%d | Path: %s",
                        episode, avg_reward, success_rate, reward, steps, action_path);
attroff(COLOR_PAIR(COLOR_HILITE));
refresh();
if (success_rate > 80.0f && episode > 1000) {
attron(COLOR_PAIR(COLOR_GOAL));
mvprintw(17, 0, "HIGH SUCCESS RATE ACHIEVED! Stopping early.");
attroff(COLOR_PAIR(COLOR_GOAL));
refresh();
sleep(2);
break;
                }
            }
        }
        trained = 1;
attron(COLOR_PAIR(COLOR_BOLD));
mvprintw(18, 0, "TRAINING COMPLETED! Avg Reward: %.1f | Success rate: %.1f%%",
                total_training_reward / TRAINING_EPISODES, (float)successes / TRAINING_EPISODES * 100.0f);
attroff(COLOR_PAIR(COLOR_BOLD));
refresh();
sleep(2);
    }
// Save model
if (save_path && trained) {
int ok = Paracast_SaveJSON(net, (char*)save_path);
attron(ok ? COLOR_PAIR(COLOR_GOAL) : COLOR_PAIR(COLOR_ERROR));
mvprintw(19, 0, ok ? "Model saved to %s" : "Save failed: %s", save_path);
attroff(ok ? COLOR_PAIR(COLOR_GOAL) : COLOR_PAIR(COLOR_ERROR));
refresh();
sleep(1);
    }
// DEMO PHASE
clear();
attron(COLOR_PAIR(COLOR_BOLD));
mvprintw(0, 0, "=== DEMO MODE ===");
mvprintw(1, 0, "Running %d demonstration episodes...", DEMO_EPS);
attroff(COLOR_PAIR(COLOR_BOLD));
refresh();
float total_demo_reward = 0.0f;
int total_demo_steps = 0;
int demo_successes = 0;
for (int ep = 0; ep < DEMO_EPS; ep++) {
int steps;
char path[64];
float reward = run_episode(net, 0, use_gpu, -1, &steps, path, sizeof(path));
        total_demo_reward += reward;
        total_demo_steps += steps;
if (reward > 50.0f) demo_successes++;
attron(COLOR_PAIR(COLOR_HILITE));
mvprintw(20 + ep, 0, "Demo %d: Reward=%6.1f | Steps=%2d | Path: %s",
                ep + 1, reward, steps, path);
attroff(COLOR_PAIR(COLOR_HILITE));
refresh();
usleep(500000); // Half second between demos
    }
// Final results
attron(COLOR_PAIR(COLOR_BOLD));
mvprintw(26, 0, "=== FINAL RESULTS ===");
mvprintw(27, 0, "Average Reward: %.1f", total_demo_reward / DEMO_EPS);
mvprintw(28, 0, "Average Steps: %.1f", (float)total_demo_steps / DEMO_EPS);
mvprintw(29, 0, "Success Rate: %.1f%%", (float)demo_successes / DEMO_EPS * 100.0f);
mvprintw(31, 0, "Press any key to exit...");
attroff(COLOR_PAIR(COLOR_BOLD));
refresh();
getch();
// Cleanup
endwin();
Paracast_Free(net);
printf("Grid RL training completed successfully!\n");
return 0;
}