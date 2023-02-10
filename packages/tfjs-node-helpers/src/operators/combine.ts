import { model as tfModel } from '@tensorflow/tfjs-node';
import { AbstractTask } from '../core/abstract-task';
import { CombinedTask } from '../core/combined-task';
import { Feature } from '../core/feature';

type BaseOptions = {
  tasks: Array<AbstractTask>;
};

export type CollapseOptions = BaseOptions & {
  mode: 'collapse';
};

export type MergeOptions = BaseOptions & {
  mode: 'merge';
};

export type RetargetOptions = BaseOptions & {
  mode: 'retarget';
  outputFeatures?: Array<Feature>;
};

export type CombineOptions = CollapseOptions | MergeOptions | RetargetOptions;

export function combine(options: CombineOptions): CombinedTask {
  // TODO: implement
  const model = tfModel({
    inputs: [],
    outputs: []
  });
  const lastSymbolicTensor = options.tasks[0].lastSymbolicTensor;
  const inputFeatures: Array<Feature> = [];
  const outputFeatures: Array<Feature> = [];

  if (options.mode === 'collapse') {
    model.getLayer().output;
  }

  return new CombinedTask({
    inputFeatures,
    outputFeatures,
    model,
    lastSymbolicTensor
  });
}
