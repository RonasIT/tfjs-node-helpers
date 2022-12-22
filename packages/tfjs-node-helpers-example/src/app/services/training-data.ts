import {
  extractFeatures,
  normalizeFeatures,
  Sample,
  splitSamplesIntoTrainingValidationTestForBinaryClassification
} from '@ronas-it/tfjs-node-helpers';
import { AgeFeatureExtractor } from '../feature-extractors/age';
import { AnnualSalaryFeatureExtractor } from '../feature-extractors/annual-salary';
import { GenderFeatureExtractor } from '../feature-extractors/gender';
import { OwnsTheCarFeatureExtractor } from '../feature-extractors/owns-the-car';
import dataset from '../../assets/data.json';
import { AgeMinMaxFeatureNormalizer } from '../feature-normalizers/age';
import { AnnualSalaryMinMaxFeatureNormalizer } from '../feature-normalizers/annual-salary';

export class TrainingDataService {
  private simulatedDelayMs: number;
  private trainingSamples: Array<Sample>;
  private validationSamples: Array<Sample>;
  private testingSamples: Array<Sample>;

  constructor({ simulatedDelayMs }: { simulatedDelayMs: number }) {
    this.simulatedDelayMs = simulatedDelayMs;
  }

  public async initialize(): Promise<void> {
    const extracts = await extractFeatures({
      data: dataset,
      inputFeatureExtractors: [
        new AgeFeatureExtractor(),
        new AnnualSalaryFeatureExtractor(),
        new GenderFeatureExtractor()
      ],
      outputFeatureExtractor: new OwnsTheCarFeatureExtractor()
    });

    const samples = await normalizeFeatures({
      extracts,
      inputFeatureNormalizers: [
        new AgeMinMaxFeatureNormalizer(),
        new AnnualSalaryMinMaxFeatureNormalizer()
      ]
    });

    const { trainingSamples, validationSamples, testingSamples } = splitSamplesIntoTrainingValidationTestForBinaryClassification(samples);

    this.trainingSamples = trainingSamples;
    this.validationSamples = validationSamples;
    this.testingSamples = testingSamples;
  }

  public async getTrainingSamples(skip: number, take: number): Promise<Array<Sample>> {
    await this.simulateDelay();

    return this.trainingSamples.slice(skip, skip + take);
  }

  public async getValidationSamples(skip: number, take: number): Promise<Array<Sample>> {
    await this.simulateDelay();

    return this.validationSamples.slice(skip, skip + take);
  }

  public async getTestingSamples(skip: number, take: number): Promise<Array<Sample>> {
    await this.simulateDelay();

    return this.testingSamples.slice(skip, skip + take);
  }

  public async getValidationSamplesCount(): Promise<number> {
    await this.simulateDelay();

    return this.validationSamples.length;
  }

  public async getTestingSamplesCount(): Promise<number> {
    await this.simulateDelay();

    return this.testingSamples.length;
  }

  private async simulateDelay(): Promise<void> {
    if (this.simulatedDelayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.simulatedDelayMs));
    }
  }
}
