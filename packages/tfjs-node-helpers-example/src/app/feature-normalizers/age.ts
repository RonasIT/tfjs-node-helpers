import { MinMaxFeatureNormalizer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';

export class AgeMinMaxFeatureNormalizer extends MinMaxFeatureNormalizer<FeatureType> {
  public featureType = FeatureType.AGE;

  constructor() {
    super({ min: 18, max: 63 });
  }
}
