import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';
import { PrecisionMetricCalculator } from './precision';
import { RecallMetricCalculator } from './recall';

export class FBetaScoreMetricCalculator extends MetricCalculator {
  constructor(private beta: number) {
    super();
  }

  public calculate(testingResult: TestingResult): Metric {
    const precision = (new PrecisionMetricCalculator()).calculate(testingResult).value;
    const recall = (new RecallMetricCalculator()).calculate(testingResult).value;

    return new Metric({
      title: `F${this.beta} Score`,
      value: ((1 + this.beta ** 2) * precision * recall) / (this.beta ** 2 * precision + recall)
    });
  }
}
