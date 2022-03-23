package com.github.chen0040.art.cluster;


import com.github.chen0040.art.ART1;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.transforms.Standardization;

import java.util.HashSet;
import java.util.Set;


/**
 * Created by xschen on 21/8/15.
 */
public class ART1Clustering {
    private ART1 net;
    private final int initialNodeCount = 1;

    
    private final int maxClusterCount = -1;

    
    private final boolean allowNewNodeInPrediction = false;
    private Standardization inputNormalization;


    /** choice parameter */
    private final double alpha = 0.1;

    /** base resonance threshold */
    private final double rho0 = 0.9;

    /** learning rate */
    private final double beta = 0.3;

    private final Set<Integer> clusterIds = new HashSet<>();

    public DataFrame fitAndTransform(DataFrame batch) {
        clusterIds.clear();
        batch = batch.makeCopy();

        int dimension = batch.row(0).toArray().length;
        inputNormalization = new Standardization(batch);

        net=new ART1(dimension * 2, initialNodeCount);
        net.setAlpha(alpha);
        net.setBeta(beta);
        net.setRho(rho0);

        int m = batch.rowCount();
        boolean create_new = true;
        for(int i=0; i < m; ++i) {
            DataRow tuple = batch.row(i);
            int clusterId = simulate(tuple, create_new);
            if(maxClusterCount > 0 && !clusterIds.contains(clusterId) && clusterIds.size() >= maxClusterCount-1){
                create_new = false;
            }
            clusterIds.add(clusterId);
        }

        return batch;
    }

    public int simulate(DataRow tuple, boolean can_create_new_node){
        double[] x = tuple.toArray();
        x = inputNormalization.standardize(x);
        double[] y = binarize(x);
        int clusterId = net.simulate(y, can_create_new_node);

        tuple.setCategoricalTargetCell("cluster", String.format("%d", clusterId));
        return clusterId;
    }

    private static double[] binarize(double[] x){
        double[] y = new double[x.length * 2];
        for(int i=0; i < x.length; ++i){
            if(x[i] > 0){
                y[i] = 1;
                y[i + x.length] = 0;
            }else{
                y[i] = 0;
                y[i + x.length] = 1;
            }
        }

        return y;
    }

}