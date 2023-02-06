import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class FDRMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const { tp, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'FDR',
      value: fp / (tp + fp)
    });
  }
}
