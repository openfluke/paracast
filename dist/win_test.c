// dist/win_test.c
#include "paracast.h"  // this header is produced next to the DLL
#include <stdio.h>
#include <string.h>

int main(void){
  int layers[] = {28,28, 10,1};
  const char* acts[] = {"relu","softmax"};
  unsigned char full[] = {1,1};

  long long h = Paracast_NewNetwork(layers, 2, (char**)acts, 2, full, 2, 0);
  if(!h){ puts("new failed"); return 1; }

  float in[28*28]; memset(in, 0, sizeof(in));
  in[0] = 1.0f;

  if(!Paracast_Forward(h, in, 28, 28)){ puts("forward failed"); return 1; }

  float out[10] = {0};
  int n = Paracast_GetOutput(h, out, 10);
  printf("n=%d out0=%f\n", n, out[0]);

  Paracast_Free(h);
  return 0;
}
