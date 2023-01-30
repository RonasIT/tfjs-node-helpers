import { Metric } from './metric';

export abstract class MetricCalculator {
  public abstract calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric;
}
