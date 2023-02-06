import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class SpecificityMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const { tn, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'Specificity',
      value: tn / (tn + fp)
    });
  }
}
