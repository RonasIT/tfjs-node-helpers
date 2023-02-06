import { Metric } from './metric';
import { MetricCalculator } from './metric-calculator';
import { TestingResult } from './result';

export const calculateMetrics = ({
  testingResult,
  metricCalculators
}: {
  testingResult: TestingResult;
  metricCalculators: Array<MetricCalculator>;
}): Array<Metric> => metricCalculators.map((calculator) => calculator.calculate(testingResult));
