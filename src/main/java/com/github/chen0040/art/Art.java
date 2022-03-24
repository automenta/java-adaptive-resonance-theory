package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/** alternative Art impl from https://github.com/thanhld94/Reinforcement-Learning
 *
 * TODO is this ART1 or ART2?
 * */
public class Art {

	public static final double RO = 0.9;
	public static final double ALPHA = 0.1;

	private int noCategories;
	private int vectorSize;
	private ArrayList <double[]> weight;
	private ArrayList <Boolean> uncommitedNode;
	private ArrayList <DoubleIntPair> choiceVector;


	public Art( int vSize ) {
		vectorSize = vSize;
		noCategories = 0;
		weight = new ArrayList<>();
		uncommitedNode = new ArrayList<>();
		choiceVector = new ArrayList<>();
		addUncommitedNode();
	}

	public int learn( double[] normalizedInput ) {
		calChoiceVector( normalizedInput );
		while ( true ) {
			/*System.out.print( "Input = " );
			for ( int i = 0; i < normalizedInput.length; i++ )
				System.out.printf( "%5.2f ", normalizedInput[ i ] );
			System.out.println();*/

			Collections.sort( choiceVector );
			int category = choiceVector.get( 0 ).getCategory();
			if ( vigilanceTest( normalizedInput, weight.get( category ) ) >= RO ) {
				if ( uncommitedNode.get( category ) ) {
					uncommitedNode.set( category, false );
					addUncommitedNode();
				}
				weight.set( category, fuzzyAnd( weight.get( category ), normalizedInput ) );
				//System.out.println( "-> Category = " + category );
				return category;
			} else {
				choiceVector.get( 0 ).reset();
			}
		}
	}

	/***************************
	*          PRIVATE         *  
	****************************/

	private Double vigilanceTest( double[] i, double[] w ) {
		Double result = l1Norm( fuzzyAnd( i, w ) ) / l1Norm( i );
		return result;
	}

	private void calChoiceVector( double[] input ) {
		for ( int j = 0; j < noCategories; j++ ) {
			choiceVector.get( j ).setCategory( j );
			if ( uncommitedNode.get( j ) )
				choiceVector.get( j ).setVal( (1.0) * vectorSize / ( ALPHA + 2 * vectorSize ) );
			else
				choiceVector.get( j ).setVal( l1Norm( fuzzyAnd( input, weight.get( j ) ) ) / ( ALPHA + l1Norm( weight.get( j ) ) ) );
		}
	}

	private Double l1Norm( double[] vector ) {
		Double result = 0.0;
		for ( int i = 0; i < vector.length; i++ )
			result += vector[ i ];
		return result;
	}

	private double[] fuzzyAnd( double[] v1, double[] v2 ) {
		//System.out.println( "Fuzzy, length = " + v1.length + " " + v2.length );
		double[] result = new double[ v1.length ];
		for ( int i = 0; i < result.length; i++ ) 
			result[ i ] = min( v1[ i ], v2[ i ] );
		return result;
	}

	private double min( double a, double b ) {
		if ( a < b ) return a;
		return b;
	}

	private void addUncommitedNode() {
		noCategories++;
		uncommitedNode.add( true );
		choiceVector.add( new DoubleIntPair() );

		double[] w = new double[ vectorSize ];
		Arrays.fill(w, 1);
		weight.add( w );
	}


	static class DoubleIntPair implements Comparable <DoubleIntPair> {

		public DoubleIntPair() {
			value = -1;
			index = -1;
		}

		public DoubleIntPair( double val, int idx ) {
			value = val;
			index = idx;
		}

		public double getVal() {
			return value;
		}

		public int getCategory() {
			return index;
		}

		public void setVal( double val ) {
			value = val;
		}

		public void setCategory( int j ) {
			index = j;
		}

		public void reset() {
			value = -1;
		}

		@Override
		public int compareTo( DoubleIntPair other ) {
			if ( other.value > this.value ) return 1;
			if ( other.value < this.value ) return -1;
			return 0;
		}

		private double value;
		private int index;
	}
}