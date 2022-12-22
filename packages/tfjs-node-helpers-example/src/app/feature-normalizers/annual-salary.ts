import { MinMaxFeatureNormalizer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';

export class AnnualSalaryMinMaxFeatureNormalizer extends MinMaxFeatureNormalizer<FeatureType> {
  public featureType = FeatureType.ANNUAL_SALARY;

  constructor() {
    super({ min: 15000, max: 152500 });
  }
}
