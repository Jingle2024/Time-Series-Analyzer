import { useCallback } from 'react';
import { useAppState } from './useAppState';
import { getHierarchyChildren } from '../services/api';

export function useHierarchy() {
  const { state, update } = useAppState();
  const { hierLevels, hierLevelValues, hierTree, hierTreeDeferred, hierChildCache, token } = state;

  function getChildValuesFromTree(path, levelIdx) {
    let node = hierTree;
    for (let i = 0; i < levelIdx; i++) {
      const v = path[hierLevels[i]];
      if (!v || !node || !node[v]) return [];
      node = node[v];
    }
    if (typeof node === 'object' && !Array.isArray(node)) return Object.keys(node);
    return [];
  }

  function childCacheKey(path, levelIdx) {
    return `${hierLevels[levelIdx] || levelIdx}::${JSON.stringify(path || {})}`;
  }

  const fetchChildValues = useCallback(async (path, levelIdx) => {
    if (levelIdx === 0) return hierLevelValues[hierLevels[0]] || [];
    if (hierTree && !hierTreeDeferred) return getChildValuesFromTree(path, levelIdx);

    const key = childCacheKey(path, levelIdx);
    if (hierChildCache[key]) return hierChildCache[key];

    const r = await getHierarchyChildren({
      token,
      path,
      next_level: hierLevels[levelIdx],
      max_values: 500,
    });
    const values = r.values || [];
    update({ hierChildCache: { ...hierChildCache, [key]: values } });
    return values;
  }, [hierLevels, hierLevelValues, hierTree, hierTreeDeferred, hierChildCache, token, update]);

  const updateNodePath = useCallback((path) => {
    update({ currentNodePath: path });
  }, [update]);

  return { fetchChildValues, updateNodePath };
}
