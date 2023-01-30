import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class FPRMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tn, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'FPR',
      value: fp / (fp + tn)
    });
  }
}
