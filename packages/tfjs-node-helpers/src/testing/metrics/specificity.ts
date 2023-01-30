import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class SpecificityMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tn, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'Specificity',
      value: tn / (tn + fp)
    });
  }
}
