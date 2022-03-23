package com.github.chen0040.art.classify;


import com.github.chen0040.art.ARTMAP;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.transforms.ComplementaryCoding;



/**
 * Created by xschen on 23/8/15.
 */
//@Getter
//@Setter
public class ARTMAPClassifier {

    //@Setter(AccessLevel.NONE)
    private ARTMAP net;

    public double alpha = 0.1; // choice parameter
    public double rho0 = 0.1; // base resonance threshold
    public double beta = 0.3; // learning rate

    private ComplementaryCoding inputNormalization;

    //@Setter(AccessLevel.NONE)
    private final boolean allowNewNodeInPrediction = false;

    public String transform(DataRow tuple) {
        return simulate(tuple, false);
    }

    public void fit(DataFrame batch) {


        inputNormalization = new ComplementaryCoding(batch);
        int dimension = batch.row(0).toArray().length * 2; // times 2 due to complementary coding

        net=new ARTMAP(dimension);
        net.setAlpha(alpha);
        net.setBeta(beta);
        net.setRho(rho0);

        int m = batch.rowCount();
        for(int i=0; i < m; ++i) {
            DataRow tuple = batch.row(i);
            simulate(tuple, true);
        }

    }

    public String simulate(DataRow tuple, boolean can_create_node){
        double[] x = tuple.toArray();
        x = inputNormalization.normalize(x);
        return net.simulate(x, tuple.categoricalTarget(), can_create_node);
    }


    public int nodeCount() {
        return net.nodeCount();
    }
}