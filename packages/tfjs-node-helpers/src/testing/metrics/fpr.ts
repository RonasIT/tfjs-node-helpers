import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class FPRMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const { tn, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'FPR',
      value: fp / (fp + tn)
    });
  }
}
