package com.github.chen0040.art.rl.minefield.agents;

import com.github.chen0040.art.rl.FalconConfig;
import com.github.chen0040.art.rl.minefield.env.MineField;
import com.github.chen0040.art.rl.q.QValue;
import com.github.chen0040.art.rl.q.QValueProvider;
import com.github.chen0040.art.rl.td.TDFalcon;
import com.github.chen0040.art.rl.td.TDLambdaFalcon;
import com.github.chen0040.art.rl.td.TDMethod;

import java.util.Set;

/**
 * Created by chen0469 on 9/29/2015 0029.
 */
public class TDFalconNavAgent extends FalconNavAgent {
    private TDFalcon ai;
    public boolean useImmediateRewardAsQ;

    public TDFalconNavAgent(FalconConfig config, int id, int numSonarInput, int numAVSonarInput, int numBearingInput, int numRangeInput) {
        super(id, numSonarInput, numAVSonarInput, numBearingInput, numRangeInput);
        ai = new TDFalcon(config);
    }

    public TDFalconNavAgent(FalconConfig config, int id, TDMethod method, int numSonarInput, int numAVSonarInput, int numBearingInput, int numRangeInput) {
        super(id, numSonarInput, numAVSonarInput, numBearingInput, numRangeInput);
        ai = new TDFalcon(config, method);
    }

    public void decayQEpsilon() {
        ai.decayQEpsilon();
    }

    @Override
    public void learn(final MineField maze) {
        Set<Integer> feasibleActionAtNewState = getFeasibleActions(maze);
        ai.learnQ(state, actions, newState, feasibleActionAtNewState, reward, createQInject(maze));
    }

    protected QValueProvider createQInject(final MineField maze) {
        return new QValueProvider() {
            public QValue queryQValue(double[] state1, int actionTaken, boolean isNextAction) {
                if (useImmediateRewardAsQ) {
                    return new QValue(reward);
                } else {
                    if (isNextAction) {
                        if (maze.willHitMine(getId(), actionTaken - 2)) {
                            return new QValue(0.0);
                        } else if (maze.willHitTarget(getId(), actionTaken - 2)) {
                            return new QValue(1.0);
                        }
                    } else {
                        if (maze.isHitMine(getId())) {
                            return new QValue(0.0);
                        } else if (maze.isHitTarget(getId())) {
                            return new QValue(1.0); //case reach target
                        }
                    }

                    return QValue.Invalid();
                }
            }
        };
    }


    @Override
    public int selectValidAction(final MineField maze) {
        Set<Integer> feasibleActions = getFeasibleActions(maze);
        return ai.selectActionId(state, feasibleActions, createQInject(maze));
    }

    @Override
    public int getNodeCount(){
        return ai.nodes.size();
    }

    public void setQGamma(double QGamma) {
        this.ai.QGamma = QGamma;
    }

    public void enableEligibilityTrace(){
        this.ai = new TDLambdaFalcon(ai.config, ai.method);
    }

}