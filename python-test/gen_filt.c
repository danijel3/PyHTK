/* Most of the code from this small program has been copied and sligtly modified
 * from the HTK project. Its copyright included below:
 */
/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Speech Vision and Robotics group                       */
/*      Cambridge University Engineering Department            */
/*      http://svr-www.eng.cam.ac.uk/                          */
/*                                                             */
/*      Entropic Cambridge Research Laboratory                 */
/*      (now part of Microsoft)                                */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*              2001  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*      File: HSigP.c:   Signal Processing Routines            */
/* ----------------------------------------------------------- */
char *hsigp_version = "!HVER!HSigP:   3.4.1 [CUED 12/03/09]";
char *hsigp_vc_id = "$Id: HSigP.c,v 1.1.1.1 2006/10/11 09:54:58 jal58 Exp $";

/*
 * While it's not strictly allowed to redistribute HTK code this way,
 * this tiny portion is used only to regenerate a bug in the code and is
 * a very tiny and insignificant part of the whole system.
 *
 * If anyone feels this doesn't constitute fair use, I will remove this file
 * from the repo immediately.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct{
    int frameSize;       /* speech frameSize */
    int numChans;        /* number of channels */
    long sampPeriod;     /* sample period */
    int fftN;            /* fft size */
    int klo,khi;         /* lopass to hipass cut-off fft indices */
    float fres;          /* scaled fft resolution */
    float* cf;           /* array[1..pOrder+1] of centre freqs */
    short* loChan;     /* array[1..fftN/2] of loChan index */
    float* loWt;         /* array[1..fftN/2] of loChan weighting */
    float* x;            /* array[1..fftN] of fftchans */
}FBankInfo;

/* EXPORT->Mel: return mel-frequency corresponding to given FFT index */
float Mel(int k,float fres)
{
    return 1127 * log(1 + (k-1)*fres);
}

/**
 * Compilation:
 * gcc gen_filt.c -o gen_filt -lm
 */

int main(int argc, char** argv)
{

    if(argc!=2)
    {
        printf("This program generates a set of triangular filter weights\n");
        printf("using the code present in HTK. For more information, please\n");
        printf("refer to the notebook explaining all this in more detail.\n");
        printf("USAGE:\n");
        printf("  ./gen_filt [number of chanels]\n");
        return 0;
    }

    int frameSize=400;
    long sampPeriod=625;
    int numChans=atoi(argv[1]);
    float lopass=80;
    float hipass=7500;

   FBankInfo fb;
   float mlo,mhi,ms,melk;
   int k,chan,maxChan,Nby2;

   /* Save sizes to cross-check subsequent usage */
   fb.frameSize = frameSize; fb.numChans = numChans;
   fb.sampPeriod = sampPeriod; 
   /* Calculate required FFT size */
   fb.fftN = 2;   
   while (frameSize>fb.fftN) fb.fftN *= 2;
   Nby2 = fb.fftN / 2;
   fb.fres = 1.0E7/(sampPeriod * fb.fftN * 700.0);
   maxChan = numChans+1;
   /* set lo and hi pass cut offs if any */
   fb.klo = 2; fb.khi = Nby2;       /* apply lo/hi pass filtering */
   mlo = 0; mhi = Mel(Nby2+1,fb.fres);
   if (lopass>=0.0) {
      mlo = 1127*log(1+lopass/700.0);
      fb.klo = (int) ((lopass * sampPeriod * 1.0e-7 * fb.fftN) + 2.5);
      if (fb.klo<2) fb.klo = 2;
   }
   if (hipass>=0.0) {
      mhi = 1127*log(1+hipass/700.0);
      fb.khi = (int) ((hipass * sampPeriod * 1.0e-7 * fb.fftN) + 0.5);
      if (fb.khi>Nby2) fb.khi = Nby2;
   }
   /* Create vector of fbank centre frequencies */
   fb.cf = (float*)malloc(sizeof(float)*(maxChan+1));
   ms = mhi - mlo;
   for (chan=1; chan <= maxChan; chan++) {
      fb.cf[chan] = ((float)chan/(float)maxChan)*ms + mlo;
   }
   
   /* Create loChan map, loChan[fftindex] -> lower channel index */
   fb.loChan = (short*)malloc(sizeof(short)*(Nby2+1));
   for (k=1,chan=1; k<=Nby2; k++){
      melk = Mel(k,fb.fres);
      if (k<fb.klo || k>fb.khi) fb.loChan[k]=-1;
      else {
         while (fb.cf[chan] < melk  && chan<=maxChan) ++chan;
         fb.loChan[k] = chan-1;
      }
   }

   /* Create vector of lower channel weights */   
   fb.loWt = (float*)malloc(sizeof(float)*(Nby2+1));
   for (k=1; k<=Nby2; k++) {
      chan = fb.loChan[k];
      if (k<fb.klo || k>fb.khi) fb.loWt[k]=0.0;
      else {
         if (chan>0) 
            fb.loWt[k] = ((fb.cf[chan+1] - Mel(k,fb.fres)) / 
                          (fb.cf[chan+1] - fb.cf[chan]));
         else
            fb.loWt[k] = (fb.cf[1]-Mel(k,fb.fres))/(fb.cf[1] - mlo);
      }
   }

    for(k=1;k<=Nby2; k++) {
        printf("%f,%d\n",fb.loWt[k],fb.loChan[k]);
    }
}
