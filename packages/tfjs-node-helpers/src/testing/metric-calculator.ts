import { Metric } from './metric';
import { TestingResult } from './result';

export abstract class MetricCalculator {
  public abstract calculate(testingResult: TestingResult): Metric;
}
