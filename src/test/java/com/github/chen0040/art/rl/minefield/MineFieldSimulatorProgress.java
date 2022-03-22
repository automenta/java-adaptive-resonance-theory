package com.github.chen0040.art.rl.minefield;

import com.github.chen0040.art.rl.minefield.env.MineField;

/**
 * Created by chen0469 on 10/2/2015 0002.
 */
public class MineFieldSimulatorProgress {
    private final MineField mineField;
    private final int trial;
    private final int run;
    private final int step;

    public MineFieldSimulatorProgress(int run, int trial, int step, MineField mineField){
        this.run = run;
        this.trial = trial;
        this.step = step;
        this.mineField = mineField;
    }

    public MineField getMineField() {
        return mineField;
    }

    public int getTrial() {
        return trial;
    }

    public int getRun() {
        return run;
    }

    public int getStep() {
        return step;
    }
}