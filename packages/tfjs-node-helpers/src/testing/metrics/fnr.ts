import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class FNRMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tp, fn } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'FNR',
      value: fn / (fn + tp)
    });
  }
}
