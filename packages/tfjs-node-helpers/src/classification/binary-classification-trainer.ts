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
import { FeatureEngineer } from '../feature-engineering/feature-engineer';
import { prepareDatasetsForBinaryClassification } from '../feature-engineering/prepare-datasets-for-binary-classification';
import { ConfusionMatrix } from '../testing/confusion-matrix';
import { Metrics } from '../testing/metrics';
import { binarize } from '../utils/binarize';

export type BinaryClassificationTrainerOptions = {
  batchSize?: number;
  epochs?: number;
  patience?: number;
  inputFeatureEngineers?: Array<FeatureEngineer<any, any>>;
  outputFeatureEngineer?: FeatureEngineer<any, any>;
  model?: LayersModel;
  hiddenLayers?: Array<layers.Layer>;
  optimizer?: string | Optimizer;
  tensorBoardLogsDirectory?: string;
};

export class BinaryClassificationTrainer {
  protected batchSize: number;
  protected epochs: number;
  protected patience: number;
  protected tensorBoardLogsDirectory?: string;
  protected inputFeatureEngineers?: Array<FeatureEngineer<any, any>>;
  protected outputFeatureEngineer?: FeatureEngineer<any, any>;
  protected model!: LayersModel;

  protected static DEFAULT_BATCH_SIZE: number = 32;
  protected static DEFAULT_EPOCHS: number = 1000;
  protected static DEFAULT_PATIENCE: number = 20;

  constructor(options: BinaryClassificationTrainerOptions) {
    this.batchSize = options.batchSize ?? BinaryClassificationTrainer.DEFAULT_BATCH_SIZE;
    this.epochs = options.epochs ?? BinaryClassificationTrainer.DEFAULT_EPOCHS;
    this.patience = options.patience ?? BinaryClassificationTrainer.DEFAULT_PATIENCE;
    this.tensorBoardLogsDirectory = options.tensorBoardLogsDirectory;
    this.inputFeatureEngineers = options.inputFeatureEngineers;
    this.outputFeatureEngineer = options.outputFeatureEngineer;

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
    metrics: Metrics;
  }> {
    const callbacks = [];

    if (this.patience !== undefined) {
      callbacks.push(tensorflowCallbacks.earlyStopping({
        patience: this.patience
      }));
    }

    if (this.tensorBoardLogsDirectory !== undefined) {
      callbacks.push(node.tensorBoard(this.tensorBoardLogsDirectory));
    }

    if (
      trainingDataset === undefined ||
      validationDataset === undefined ||
      testingDataset === undefined
    ) {
      if (this.inputFeatureEngineers === undefined || this.outputFeatureEngineer === undefined) {
        throw new Error('trainingDataset, validationDataset and testingDataset are required when inputFeatureEngineers and outputFeatureEngineer are not provided!');
      }

      const datasets = await prepareDatasetsForBinaryClassification({
        data: data as Array<any>,
        inputFeatureEngineers: this.inputFeatureEngineers,
        outputFeatureEngineer: this.outputFeatureEngineer,
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
      if (options.hiddenLayers !== undefined && options.inputFeatureEngineers !== undefined) {
        const inputLayer = input({ shape: [options.inputFeatureEngineers.length] });
        let symbolicTensor = inputLayer;

        for (const layer of options.hiddenLayers) {
          symbolicTensor = layer.apply(symbolicTensor) as SymbolicTensor;
        }

        const outputLayer = layers
          .dense({ units: 1, activation: 'sigmoid' })
          .apply(symbolicTensor) as SymbolicTensor;

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
    metrics: Metrics;
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

    const trueValues = await testYs.data<'float32'>();
    const predictedValues = await binarizedPredictions.data<'float32'>();

    const confusionMatrix = this.calculateConfusionMatrix(trueValues, predictedValues);
    const metrics = this.calculateMetrics(confusionMatrix);

    if (printTestingResults) {
      this.printTestResults(loss, confusionMatrix, metrics);
    }

    return { loss, confusionMatrix, metrics };
  }

  private calculateConfusionMatrix(trueValues: Float32Array, predictedValues: Float32Array): ConfusionMatrix {
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    for (let index = 0; index < trueValues.length; index++) {
      const trueValue = trueValues[index];
      const predictedValue = predictedValues[index];

      if (trueValue === 1 && predictedValue === 1) {
        tp++;
      } else if (trueValue === 0 && predictedValue === 0) {
        tn++;
      } else if (trueValue === 0 && predictedValue === 1) {
        fp++;
      } else {
        fn++;
      }
    }

    return { tp, tn, fp, fn };
  }

  private calculateMetrics({ fp, fn, tp, tn }: ConfusionMatrix): Metrics {
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp);
    const recall = tp / (tp + fn);
    const f1 = (2 * precision * recall) / (precision + recall);
    const specificity = tn / (tn + fp);
    const fpr = fp / (fp + tn);
    const fnr = fn / (fn + tp);

    return {
      accuracy,
      precision,
      recall,
      f1,
      specificity,
      fpr,
      fnr
    };
  }

  private printTestResults(loss: number, confusionMatrix: ConfusionMatrix, metrics: Metrics): void {
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

  private printMetricsTable(loss: number, metrics: Metrics): void {
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

    metricsTable.addRow({
      metric: 'Accuracy',
      value: metrics.accuracy.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'Precision',
      value: metrics.precision.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'Recall',
      value: metrics.recall.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'F1 Score',
      value: metrics.f1.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'Specificity',
      value: metrics.specificity.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'FPR',
      value: metrics.fpr.toFixed(4)
    });

    metricsTable.addRow({
      metric: 'FNR',
      value: metrics.fnr.toFixed(4)
    });

    metricsTable.printTable();
  }
}
