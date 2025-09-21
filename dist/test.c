#include "libparacast.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void){
  // 28x28 input, 10x1 output
  int layers[] = {28,28, 10,1};
  const char* acts[] = {"relu","softmax"};
  unsigned char full[] = {1,1}; // fully connected for both layers

  long long h = Paracast_NewNetwork(layers, 2, (char**)acts, 2, full, 2, 0);
  if(!h){ puts("new failed"); return 1; }

  float in[28*28]; memset(in, 0, sizeof(in));
  in[0] = 1.0f;

  if(!Paracast_Forward(h, in, 28, 28)){ puts("forward failed"); return 1; }

  float out[10]; memset(out, 0, sizeof(out));
  int n = Paracast_GetOutput(h, out, 10);
  printf("n=%d, out[0]=%f\n", n, out[0]);

  Paracast_Free(h);
  return 0;
}
