import {
  extractFeatures,
  Sample,
  splitSamplesIntoTrainingValidationTestForBinaryClassification
} from '@ronas-it/tfjs-node-helpers';
import { AgeFeatureEngineer } from '../feature-engineers/age';
import { AnnualSalaryFeatureEngineer } from '../feature-engineers/annual-salary';
import { GenderFeatureEngineer } from '../feature-engineers/gender';
import { OwnsTheCarFeatureEngineer } from '../feature-engineers/owns-the-car';
import dataset from '../../assets/data.json';

export class TrainingDataService {
  private simulatedDelayMs: number;
  private trainingSamples: Array<Sample>;
  private validationSamples: Array<Sample>;
  private testingSamples: Array<Sample>;

  constructor({ simulatedDelayMs }: { simulatedDelayMs: number }) {
    this.simulatedDelayMs = simulatedDelayMs;
  }

  public async initialize(): Promise<void> {
    const samples = await extractFeatures({
      data: dataset,
      inputFeatureEngineers: [
        new AgeFeatureEngineer(),
        new AnnualSalaryFeatureEngineer(),
        new GenderFeatureEngineer()
      ],
      outputFeatureEngineer: new OwnsTheCarFeatureEngineer()
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
