import { Feature } from '../core/feature';
import { Task } from '../core/task';

export type CombineOptions = {
  tasks: Array<Task>;
  mode: 'collapse' | 'merge' | 'retarget';
  outputFeatures?: Array<Feature>;
};

export function combine(options: CombineOptions): Task {
  return new Task({
    inputFeatures: [],
    outputFeatures: []
  });
}
