#include <stdio.h>
#include <math.h>

#define FREQ_RESOLUTION 500      // Number of freq points to calculate
#define ANGLE_RESOLUTION 500     // Number of angle points to calculate

int main(void)
{
   int numElements = 4;          // Number of array elements
   double spacing = 0.2;         // Element separation in metre
   double speedSound = 343.0;    // m/s

   int f, a, i;

   // Iterate through arrival angle points
   for (f=0 ; f<FREQ_RESOLUTION ; f++)
   {
      double freq = 10000.0 * f / (FREQ_RESOLUTION-1);

      for (a=0 ; a<ANGLE_RESOLUTION ; a++)
      {
         // Calculate the planewave arrival angle
         double angle = -90 + 180.0 * a / (ANGLE_RESOLUTION-1);
         double angleRad = M_PI * (double) angle / 180;

         double realSum = 0;
         double imagSum = 0;

         // Iterate through array elements
         for (i=0 ; i<numElements ; i++)
         {
            // Calculate element position and wavefront delay
            double position = i * spacing;
            double delay = position * sin(angleRad) / speedSound;

            // Add Wave
            realSum += cos(2.0 * M_PI * freq * delay);
            imagSum += sin(2.0 * M_PI * freq * delay);
         }

         double output = sqrt(realSum * realSum + imagSum * imagSum) / numElements;
         double logOutput = 20 * log10(output);
         if (logOutput < -50) logOutput = -50;
         printf("%f %f %f\n", angle, freq, logOutput);
      }

      printf("\n");
   }

   return 0;
}