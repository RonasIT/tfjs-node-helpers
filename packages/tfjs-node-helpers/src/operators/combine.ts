import { AbstractTask } from '../core/abstract-task';
import { CombinedTask } from '../core/combined-task';
import { Feature } from '../core/feature';

export type CombineOptions = {
  tasks: Array<AbstractTask>;
  mode: 'collapse' | 'merge' | 'retarget';
  outputFeatures?: Array<Feature>;
};

export function combine(options: CombineOptions): CombinedTask {
  // TODO: implement
  const model = options.tasks[0].model;
  const lastSymbolicTensor = options.tasks[0].lastSymbolicTensor;

  return new CombinedTask({
    inputFeatures: [],
    outputFeatures: [],
    model,
    lastSymbolicTensor
  });
}
