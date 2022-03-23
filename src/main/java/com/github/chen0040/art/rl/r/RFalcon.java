package com.github.chen0040.art.rl.r;

import com.github.chen0040.art.rl.Falcon;
import com.github.chen0040.art.rl.FalconConfig;
import com.github.chen0040.art.rl.FalconNode;

/**
 * Created by chen0469 on 10/1/2015 0001.
 */
public class RFalcon extends Falcon {

    private final double initConfidence = 0.5;
    private final double reinforce_rate = 0.5;
    private final double penalize_rate = 0.2;
    private final double decay_rate = 0.0005;
    private final double threshold = 0.01;
    private final int    capacity = 9999;
    private double[]   confidence;
    private int J = -1;

    public RFalcon(FalconConfig config) {
        super(config);
    }

    @Override
    public int learn(double[] state, double[] actions, double[] rewards){
        double[] choiceValues = choiceValues(nodes, state, null, null, config);
        return learn(state, actions, rewards, choiceValues);
    }

    @Override
    protected void onNewNode(FalconNode node){
        int n = nodes.size();

        double[] old_confidence = this.confidence;
        double[] new_confidence = new double[n];
        for (int j=0; j < n-1; j++)
            new_confidence[j] = old_confidence[j]; //TODO j+-1 on one of these indices?
        new_confidence[n-1] = initConfidence;
        this.confidence = new_confidence;

        J = n-1;
    }

    @Override
    public void onChoiceCompeted(int J){
        this.J = J;
    }

    public void reinforce () {
        confidence[J] += (1.0-confidence[J]) * reinforce_rate;
    }

    public void penalize () {
        confidence[J] -= confidence[J] * penalize_rate;
    }

    public void decay () {
        int numCode = nodes.size();
        for (int j=0; j < numCode; j++)
            confidence[j] -= confidence[j] * decay_rate;
    }
}