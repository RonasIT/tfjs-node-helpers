import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class PrecisionMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tp, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'Precision',
      value: tp / (tp + fp)
    });
  }
}
