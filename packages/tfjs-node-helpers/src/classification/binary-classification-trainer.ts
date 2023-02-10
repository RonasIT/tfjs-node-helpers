import {
  callbacks as tensorflowCallbacks,
  data,
  input,
  layers,
  LayersModel,
  model,
  node,
  Optimizer,
  Scalar,
  SymbolicTensor,
  Tensor,
  TensorContainer
} from '@tensorflow/tfjs-node';
import { green, red } from 'chalk';
import { Table } from 'console-table-printer';
import { FeatureExtractor } from '../feature-engineering/feature-extractor';
import { prepareDatasetsForBinaryClassification } from '../feature-engineering/prepare-datasets-for-binary-classification';
import { calculateMetrics } from '../testing/calculate-metrics';
import { ConfusionMatrix } from '../testing/confusion-matrix';
import { Metric } from '../testing/metric';
import { MetricCalculator } from '../testing/metric-calculator';
import { binarize } from '../utils/binarize';
import { FeatureNormalizer } from '../feature-engineering/feature-normalizer';
import { switchHardwareUsage } from '../utils/switch-hardware-usage';

export type BinaryClassificationTrainerOptions = {
  shouldUseGPU?: boolean;
  batchSize?: number;
  epochs?: number;
  patience?: number;
  inputFeatureExtractors?: Array<FeatureExtractor<any, any>>;
  outputFeatureExtractor?: FeatureExtractor<any, any>;
  inputFeatureNormalizers?: Array<FeatureNormalizer<any>>;
  model?: LayersModel;
  hiddenLayers?: Array<layers.Layer>;
  optimizer?: string | Optimizer;
  tensorBoardLogsDirectory?: string;
  metricCalculators?: Array<MetricCalculator>;
};

export class BinaryClassificationTrainer {
  protected batchSize: number;
  protected epochs: number;
  protected patience: number;
  protected tensorBoardLogsDirectory?: string;
  protected inputFeatureExtractors?: Array<FeatureExtractor<any, any>>;
  protected outputFeatureExtractor?: FeatureExtractor<any, any>;
  protected inputFeatureNormalizers?: Array<FeatureNormalizer<any>>;
  protected model!: LayersModel;
  protected metricCalculators: Array<MetricCalculator>;

  protected static DEFAULT_BATCH_SIZE: number = 32;
  protected static DEFAULT_EPOCHS: number = 1000;
  protected static DEFAULT_PATIENCE: number = 20;

  constructor(options: BinaryClassificationTrainerOptions) {
    switchHardwareUsage(options.shouldUseGPU);

    this.batchSize = options.batchSize ?? BinaryClassificationTrainer.DEFAULT_BATCH_SIZE;
    this.epochs = options.epochs ?? BinaryClassificationTrainer.DEFAULT_EPOCHS;
    this.patience = options.patience ?? BinaryClassificationTrainer.DEFAULT_PATIENCE;
    this.tensorBoardLogsDirectory = options.tensorBoardLogsDirectory;
    this.inputFeatureExtractors = options.inputFeatureExtractors;
    this.outputFeatureExtractor = options.outputFeatureExtractor;
    this.inputFeatureNormalizers = options.inputFeatureNormalizers;
    this.metricCalculators = options.metricCalculators || [];

    this.initializeModel(options);
  }

  public async trainAndTest({
    data,
    trainingDataset,
    validationDataset,
    testingDataset,
    printTestingResults
  }: {
    data?: Array<any>;
    trainingDataset?: data.Dataset<TensorContainer>;
    validationDataset?: data.Dataset<TensorContainer>;
    testingDataset?: data.Dataset<TensorContainer>;
    printTestingResults?: boolean;
  }): Promise<{
    loss: number;
    confusionMatrix: ConfusionMatrix;
    metrics: Array<Metric>;
  }> {
    const callbacks = [];

    if (this.patience !== undefined) {
      callbacks.push(
        tensorflowCallbacks.earlyStopping({
          patience: this.patience
        })
      );
    }

    if (this.tensorBoardLogsDirectory !== undefined) {
      callbacks.push(node.tensorBoard(this.tensorBoardLogsDirectory));
    }

    if (trainingDataset === undefined || validationDataset === undefined || testingDataset === undefined) {
      if (
        this.inputFeatureExtractors === undefined ||
        this.outputFeatureExtractor === undefined ||
        this.inputFeatureNormalizers === undefined
      ) {
        throw new Error(
          'trainingDataset, validationDataset and testingDataset are required when inputFeatureExtractors, outputFeatureExtractor and inputFeatureNormalizers are not provided!'
        );
      }

      const datasets = await prepareDatasetsForBinaryClassification({
        data: data as Array<any>,
        inputFeatureExtractors: this.inputFeatureExtractors,
        outputFeatureExtractor: this.outputFeatureExtractor,
        inputFeatureNormalizers: this.inputFeatureNormalizers,
        batchSize: this.batchSize
      });

      trainingDataset = datasets.trainingDataset;
      validationDataset = datasets.validationDataset;
      testingDataset = datasets.testingDataset;
    }

    await this.model.fitDataset(trainingDataset, {
      epochs: this.epochs,
      validationData: validationDataset,
      callbacks
    });

    return await this.test({ testingDataset, printTestingResults });
  }

  public async save(path: string): Promise<void> {
    await this.model.save(`file://${path}`);
  }

  private initializeModel(options: BinaryClassificationTrainerOptions): void {
    if (options.model !== undefined) {
      this.model = options.model;
    } else {
      if (options.hiddenLayers !== undefined && options.inputFeatureExtractors !== undefined) {
        const inputLayer = input({
          shape: [options.inputFeatureExtractors.length]
        });
        let symbolicTensor = inputLayer;

        for (const layer of options.hiddenLayers) {
          symbolicTensor = layer.apply(symbolicTensor) as SymbolicTensor;
        }

        const outputLayer = layers.dense({ units: 1, activation: 'sigmoid' }).apply(symbolicTensor) as SymbolicTensor;

        this.model = model({
          inputs: inputLayer,
          outputs: outputLayer
        });
      } else {
        throw new Error('hiddenLayers and inputFeatureExtractors options are required when the model is not provided!');
      }
    }

    this.model.compile({
      optimizer: options.optimizer ?? 'adam',
      loss: 'binaryCrossentropy'
    });
  }

  private async test({
    testingDataset,
    printTestingResults
  }: {
    testingDataset: data.Dataset<TensorContainer>;
    printTestingResults?: boolean;
  }): Promise<{
    loss: number;
    confusionMatrix: ConfusionMatrix;
    metrics: Array<Metric>;
  }> {
    const lossTensor = (await this.model.evaluateDataset(testingDataset as data.Dataset<any>, {})) as Scalar;
    const [loss] = await lossTensor.data();

    const [testingData] = (await testingDataset.toArray()) as Array<{
      xs: Tensor;
      ys: Tensor;
    }>;

    const testXs = testingData.xs;
    const testYs = testingData.ys;

    const predictions = this.model.predict(testXs) as Tensor;
    const binarizedPredictions = binarize(predictions);

    const confusionMatrix = new ConfusionMatrix(testYs, binarizedPredictions);
    const metrics = calculateMetrics({
      testingResult: {
        trueValues: testYs,
        predictedValues: binarizedPredictions,
        probabilities: predictions
      },
      metricCalculators: this.metricCalculators
    });

    if (printTestingResults) {
      this.printTestResults(loss, confusionMatrix, metrics);
    }

    return { loss, confusionMatrix, metrics };
  }

  private printTestResults(loss: number, confusionMatrix: ConfusionMatrix, metrics: Array<Metric>): void {
    console.log('\n');
    this.printConfusionMatrixTable(confusionMatrix);
    console.log('\n');
    this.printMetricsTable(loss, metrics);
  }

  private printConfusionMatrixTable(confusionMatrix: ConfusionMatrix): void {
    const confusionMatrixTable = new Table({
      title: 'Confusion Matrix',
      columns: [
        { name: 'column1', title: ' ' },
        { name: 'column2', title: 'Real 0' },
        { name: 'column3', title: 'Real 1' }
      ]
    });

    confusionMatrixTable.addRow({
      column1: 'Predicted 0',
      column2: green(confusionMatrix.tn),
      column3: red(confusionMatrix.fn)
    });

    confusionMatrixTable.addRow({
      column1: 'Predicted 1',
      column2: red(confusionMatrix.fp),
      column3: green(confusionMatrix.tp)
    });

    confusionMatrixTable.printTable();
  }

  private printMetricsTable(loss: number, metrics: Array<Metric>): void {
    const metricsTable = new Table({
      title: 'Metrics',
      columns: [
        { name: 'metric', title: 'Metric' },
        { name: 'value', title: 'Value' }
      ]
    });

    metricsTable.addRow({
      metric: 'Loss',
      value: loss.toFixed(4)
    });

    metrics.forEach((metric) =>
      metricsTable.addRow({
        metric: metric.title,
        value: metric.value.toFixed(4)
      })
    );

    metricsTable.printTable();
  }
}
