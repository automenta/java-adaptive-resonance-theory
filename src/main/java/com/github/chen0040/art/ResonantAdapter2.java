package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.List;

/**
 * PtrMan's try of implementation of ART-2
 * <p>
 * translated from:
 * https://gist.github.com/PtrMan/18e467c8f3158017815beba1b319588c
 * <p>
 * DONE< implement T2->T1 weights and proper voting! >
 * DONE< add bottom up weights >
 * DONE< implement proper voting >
 * DONE< implement proper bottom up voting! >
 * DONE< apply value of prototype when a valid prototype was selected! >
 * DONE< adapt bottom-up weights! >
 * DONE< implement reset!!! >
 *
 * see: https://colab.research.google.com/drive/17kjLbN8dhQpsA6Q6CEr2b6kSHP5rosfH
 */
public class ResonantAdapter2 {

    public double a = 2; //10.0;
    public double b = 1.2;
    public double c = 0.1;

    /** some kind of learning rate: 0 < d < 1 */
    public double d = 0.8;


    public double e = 0.01;

    /** parameter for noise suppression */
    public double sigma = 0.0;

    /** in range 0 < x < 1 */
    public double vigilance = 1;

    public int vecWidth = 5; // width of vector

    public int roundsLtm = 3; // rounds of LTM

    public List<ArtPrototype> prototypes = new ArrayList<>();

    /** number of prototypes */
    public int M;

    public boolean enLearning = true; // enable learning? - is useful to toggle to pure classification


    public ResonantAdapter2(int M) {
        this.M = M;
        /*
        { // check the ratio between c and d
            var ratio: Float = (c*d) / (1.0-d);
            trace('ratio=$ratio');
            if (ratio >= 1.0) {
                throw new haxe.Exception("ratio violated!");
            }
        }

        { // check 2/3 rule inequality by ART1 paper
            var condA: Bool = Math.max(1.0, d) < b;
            var condB: Bool = b < d + 1.0;
            var cond: Bool = condA&&condB;
            trace('cond=${cond}');
            if (!cond) {
                throw new haxe.Exception("cond violated!");
            }
        }


        for(im in 0...M) {
            var createdPrototype = new ArtPrototype(vecNull(vecWidth));

            // init bottom up weights as described in the paper
            var scale: Float = 1.0 / ((1.0 - d)*Math.sqrt(M));
            createdPrototype.zBottomUp = [];
            for(ii in 0...vecWidth) {
                var v: Float = scale*0.1 + Math.random()*scale*0.1*0.5;
                createdPrototype.zBottomUp.push(v);
            }

            prototypes.push(createdPrototype);
        }
    }
    */
    }

    public int calc(double[] i) {
        throw new UnsupportedOperationException("TODO");
        /*
        // reset surpression by reset
        for(iPrototype in prototypes) {
            iPrototype.isSurpressed = false;
        }

        var selIdx: Int = -1;

        var u: Array<Float> = null;

        for (iTry in 0...5) { // loop over attemps which is reseted by reset mechanism

            selIdx = -1;
            ///if (prototypes.length > 0) {
            ///    selIdx = 0; // HACK< select first prototype until selection of prototype candidate is implemented >
            ///}

            var enReset: Bool = false;

            for (iOuterRound in 0...2) {

                var v: Array<Float> = vecNull(vecWidth);

                var p: Array<Float> = null;

                // # solver for ART STM equations
                for(iRound in 0...roundsLtm) {
                    // u_i = v_i / (e + ||v||)             (6)
                    u = vecScale(v, 1.0 / (e + vecNorm2(v)));

                    // p_i = u_i + sum_j ( g(y_i)*z_ji )   (4)
                    p = u; // special case if no category was learned yet!
                    if (selIdx != -1) { // was a candidate selected?
                        var temp0 = vecScale(prototypes[selIdx].v, d);
                        p = vecAdd(u, temp0);
                    }

                    // q_i = p_i / (e + ||p||)             (5)
                    var q = vecScale(p, 1.0 / (e + vecNorm2(p)));


                    // w_i = I_i + au_i                    (8)
                    var w = vecAdd(I, vecScale(u, a));

                    // x_i = w_i / (e + ||w||)             (9)
                    var x = vecScale(w, 1.0 / (e + vecNorm2(w)));


                    // v_i = f(x_i) + b f(q_i)             (7)
                    v = vecAdd(applyF(x), vecScale(applyF(q), b));
                }


                // # F2 matching
                if (iOuterRound == 0) { // we only match F2 if we didn't find yet a best candidate!
                    var winnerPrototypeIdx: Int = -1;
                    var winnerPrototypeTj: Float = Math.NEGATIVE_INFINITY;
                    for(iProtoIdx in 0...prototypes.length) {
                        var iPrototype: ArtPrototype = prototypes[iProtoIdx];

                        if (iPrototype.isSurpressed) {
                            continue; // ignore surpressed prototypes for F2 matching!
                        }

                        var Tj: Float = vecDot(p, iPrototype.zBottomUp);
                        if (Tj > winnerPrototypeTj) {
                            winnerPrototypeTj = Tj;
                            winnerPrototypeIdx = iProtoIdx;
                        }
                    }

                    trace('winner prototype idx=$winnerPrototypeIdx Tj=$winnerPrototypeTj');

                    selIdx = winnerPrototypeIdx;
                }



                if (iOuterRound == 1) {
                    // compute reset
                    //                                 formula (20)
                    var r = vecScale(vecAdd(u, vecScale(p, c)), 1.0 / (e + vecNorm2(u)+vecNorm2(vecScale(p,c))));

                    // reset when ever an input pattern is active and when the reset condition is true
                    enReset = vigilance / (e + vecNorm2(r)) > 1.0;
                    trace('DBG ${1.0 / (e + vecNorm2(r))}');
                    trace('enReset=$enReset');

                    // reset mechanism which surpresses a prototype if it doesn't fit
                    if (enReset) {
                        prototypes[selIdx].isSurpressed = true; // surpress by reset
                    }
                }

            }

            if (!enReset) {
                break; // we don't need another round if reset is not active!
            }
        }




        // # LTM
        if (enLearning) { // only do if learning is enabled
            { // adapt top-down weights F2->F1
                var zJ: Array<Float> = prototypes[selIdx].v;

                var temp0 = vecScale(u, 1.0 / (1.0 - d));
                temp0 = vecSub(temp0, zJ);
                var delta = vecScale(temp0, d*(1.0-d)); // compute delta to adjust weights

                zJ = vecAdd(zJ, delta); // actually update weights

                trace('zJ=');
                for(iv in zJ) {
                    trace('   $iv');
                }

                prototypes[selIdx].v = zJ; // update prototype
            }



            { // adapt bottom-up weights F1->F2
                var ziJ: Array<Float> = prototypes[selIdx].zBottomUp;
                var temp0 = vecScale(u, 1.0 / (1.0 - d));
                temp0 = vecSub(temp0, ziJ);
                var delta = vecScale(temp0, d*(1.0-d)); // compute delta to adjust weights

                ziJ = vecAdd(ziJ, delta); // actually update weights

                trace('ziJ=');
                for(iv in ziJ) {
                    trace('   $iv');
                }

                prototypes[selIdx].zBottomUp = ziJ; // update prototype
            }
        }

        return selIdx; // we return the classification
        */
    }

    public double[] applyF(double[] x) {
        throw new UnsupportedOperationException("TODO");
    /*
        var res = [];
        for (ix in x) {
            var v = ix;
            if (ix < sigma) {
                v = (2.0*sigma*ix*ix) / (ix*ix + sigma*sigma); // nonlinearity
            }
            res.push(v);
        }
        return res;
    }

    */
    }

    /*
    public static function vecScale(a: Array<Float>, s: Float): Array<Float> {
        var res: Array<Float> = [];
        for (iv in a) {
            res.push(iv*s);
        }
        return res;
    }

    public static function vecAdd(a: Array<Float>, b: Array<Float>): Array<Float> {
        var res: Array<Float> = [];
        for(iidx in 0...a.length) {
            var v: Float = a[iidx]+b[iidx];
            res.push(v);
        }
        return res;
    }

    public static function vecSub(a: Array<Float>, b: Array<Float>): Array<Float> {
        var res: Array<Float> = [];
        for(iidx in 0...a.length) {
            var v: Float = a[iidx]-b[iidx];
            res.push(v);
        }
        return res;
    }

    public static function vecNorm2(v: Array<Float>): Float {
        var l = 0.0;
        for(iv in v) {
            l += iv*iv;
        }
        return Math.sqrt(l);
    }

    public static function vecNull(width: Int): Array<Float> {
        var res = [];
        for(i in 0...width) {
            res.push(0.0);
        }
        return res;
    }

    public static function vecDot(a: Array<Float>, b: Array<Float>): Float {
        var v = 0.0;
        for(iidx in 0...a.length) {
            v += a[iidx]*b[iidx];
        }
        return v;
    }
}

*/

    static class ArtPrototype {
        public double[] v;

        /**
         * bottom up weights F1 -> F2
         */
        public double[] zBottomUp;

        /**
         * is it currently surpressed by reset?
         */
        public boolean isSurpressed = false;

        public ArtPrototype(double[] v) {
            this.v = v;
        }
    }
}