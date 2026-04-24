const COMBINATION_TYPES = {
  SINGLE: 'SINGLE',
  PAIR: 'PAIR',
  TRIPLE: 'TRIPLE',
  STRAIGHT: 'STRAIGHT',
  FULL_HOUSE: 'FULL_HOUSE',
  SEQ_3_PAIRS: 'SEQ_3_PAIRS',
  SEQ_2_TRIPLES: 'SEQ_2_TRIPLES',
  JOKER_PAIR: 'JOKER_PAIR',
  FOUR_JOKERS: 'FOUR_JOKERS',
  FOUR_OF_A_KIND: 'FOUR_OF_A_KIND',
  FIVE_OF_A_KIND: 'FIVE_OF_A_KIND',
  SIX_OF_A_KIND: 'SIX_OF_A_KIND',
  SEVEN_OF_A_KIND: 'SEVEN_OF_A_KIND',
  EIGHT_OF_A_KIND: 'EIGHT_OF_A_KIND',
  STRAIGHT_FLUSH: 'STRAIGHT_FLUSH',
  INVALID: 'INVALID'
};

const BOMB_TYPES = [
  COMBINATION_TYPES.FOUR_JOKERS,
  COMBINATION_TYPES.FOUR_OF_A_KIND,
  COMBINATION_TYPES.FIVE_OF_A_KIND,
  COMBINATION_TYPES.SIX_OF_A_KIND,
  COMBINATION_TYPES.SEVEN_OF_A_KIND,
  COMBINATION_TYPES.EIGHT_OF_A_KIND,
  COMBINATION_TYPES.STRAIGHT_FLUSH
];

const ALL_STANDARD_CARDS = [];
const suits = ['hearts', 'diamonds', 'clubs', 'spades'];
const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
for (const suit of suits) {
  for (const rank of ranks) {
    ALL_STANDARD_CARDS.push({ suit, rank });
  }
}

const getRankValue = (rank, currentRoundLevelRank = null) => {
  const rankOrder = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
  };
  
  if (currentRoundLevelRank && rank === currentRoundLevelRank) {
    return 15; // Level rank is higher than A (14)
  }
  
  return rankOrder[rank] || 0;
};

// Sequences ignore the levelRank wild aspect and use purely natural rank
const getRankValueForStraight = (rank, useAceAsOne = false) => {
  if (rank === 'A' && useAceAsOne) return 1;
  const rankOrder = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
  };
  return rankOrder[rank] || 0;
};

export const sortCardsByRank = (cards, levelRank = null) => {
  return [...cards].sort((a, b) => {
    const aIsRedJoker = a.joker === 'red';
    const bIsRedJoker = b.joker === 'red';
    const aIsBlackJoker = a.joker === 'black';
    const bIsBlackJoker = b.joker === 'black';
    
    if (aIsRedJoker && bIsRedJoker) return 0;
    if (aIsRedJoker) return 1;
    if (bIsRedJoker) return -1;
    
    if (aIsBlackJoker && bIsBlackJoker) return 0;
    if (aIsBlackJoker) return 1;
    if (bIsBlackJoker) return -1;
    
    const aIsLevelRank = levelRank && a.rank === levelRank;
    const bIsLevelRank = levelRank && b.rank === levelRank;
    
    if (aIsLevelRank && bIsLevelRank) return 0;
    if (aIsLevelRank) return 1;
    if (bIsLevelRank) return -1;
    
    const rankOrder = {
      '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
      '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    };
    
    const aValue = rankOrder[a.rank] || 0;
    const bValue = rankOrder[b.rank] || 0;
    
    return aValue - bValue;
  });
};

const countRanks = (cards) => {
  const counts = {};
  cards.forEach(card => {
    if (card.joker) return;
    const rank = card.rank;
    counts[rank] = (counts[rank] || 0) + 1;
  });
  return counts;
};

const detectFourJokers = (cards) => {
  if (cards.length !== 4) return null;
  
  const jokers = cards.filter(c => c.joker);
  if (jokers.length !== 4) return null;
  
  const redJokers = jokers.filter(c => c.joker === 'red').length;
  const blackJokers = jokers.filter(c => c.joker === 'black').length;
  
  if (redJokers === 2 && blackJokers === 2) {
    return {
      type: COMBINATION_TYPES.FOUR_JOKERS,
      rank: 1000,
      isValid: true,
      isBomb: true,
      bombStrength: 1000,
      cards: cards
    };
  }
  
  return null;
};

const detectNOfAKind = (cards, n, currentRoundLevelRank = null) => {
  if (cards.length !== n) return null;
  
  if (cards.some(c => c.joker)) return null;
  
  const rankCounts = countRanks(cards);
  const ranks = Object.keys(rankCounts);
  
  if (ranks.length !== 1) return null;
  if (rankCounts[ranks[0]] !== n) return null;
  
  const rank = ranks[0];
  const typeMap = {
    4: COMBINATION_TYPES.FOUR_OF_A_KIND,
    5: COMBINATION_TYPES.FIVE_OF_A_KIND,
    6: COMBINATION_TYPES.SIX_OF_A_KIND,
    7: COMBINATION_TYPES.SEVEN_OF_A_KIND,
    8: COMBINATION_TYPES.EIGHT_OF_A_KIND
  };
  
  if (!typeMap[n]) return null;
  
  const rankValue = getRankValue(rank, currentRoundLevelRank);
  const bombStrength = n * 100 + rankValue;
  
  return {
    type: typeMap[n],
    rank: rankValue,
    isValid: true,
    isBomb: true,
    bombStrength: bombStrength,
    bombCount: n,
    cards: cards
  };
};

const detectStraightFlush = (cards, currentRoundLevelRank = null) => {
  if (cards.length !== 5) return null;
  if (cards.some(c => c.joker)) return null;
  
  const suits = [...new Set(cards.map(c => c.suit))];
  if (suits.length !== 1) return null;
  
  const sorted = sortCardsByRank(cards, null);
  const ranks = sorted.map(c => c.rank);
  const uniqueRanks = [...new Set(ranks)];
  
  if (uniqueRanks.length !== cards.length) return null;
  
  const trySequence = (useAceAsOne) => {
    const values = sorted.map(c => getRankValueForStraight(c.rank, useAceAsOne));
    for (let i = 1; i < values.length; i++) {
      if (values[i] !== values[i - 1] + 1) return false;
    }
    if (useAceAsOne && values[values.length - 1] === 14) return false;
    if (!useAceAsOne && values[0] === 1) return false;
    return true;
  };
  
  const hasAce = ranks.includes('A');
  let highestRank = 0;
  
  if (hasAce) {
    if (trySequence(true)) {
      highestRank = getRankValueForStraight(sorted[sorted.length - 1].rank, true);
    } else if (trySequence(false)) {
      highestRank = getRankValueForStraight(sorted[sorted.length - 1].rank, false);
    } else {
      return null;
    }
  } else {
    if (trySequence(false)) {
      highestRank = getRankValueForStraight(sorted[sorted.length - 1].rank, false);
    } else {
      return null;
    }
  }
  
  const bombStrength = 900 + cards.length * 10 + highestRank;
  
  return {
    type: COMBINATION_TYPES.STRAIGHT_FLUSH,
    rank: highestRank,
    isValid: true,
    isBomb: true,
    bombStrength: bombStrength,
    flushLength: cards.length,
    cards: cards
  };
};

const detectBombBase = (cards, currentRoundLevelRank = null) => {
  const fourJokers = detectFourJokers(cards);
  if (fourJokers) return fourJokers;
  
  for (let n = 8; n >= 4; n--) {
    const nOfAKind = detectNOfAKind(cards, n, currentRoundLevelRank);
    if (nOfAKind) return nOfAKind;
  }
  
  const straightFlush = detectStraightFlush(cards, currentRoundLevelRank);
  if (straightFlush) return straightFlush;
  
  return null;
};

export const isBomb = (combination) => {
  return combination && combination.isBomb === true;
};

export const compareBombs = (bomb1, bomb2) => {
  if (!isBomb(bomb1) || !isBomb(bomb2)) return false;
  return bomb1.bombStrength > bomb2.bombStrength;
};

export const compareCombinations = (combo1, combo2, currentRoundLevelRank = null) => {
  if (!combo1.isValid || !combo2.isValid) return false;
  
  const combo1IsBomb = isBomb(combo1);
  const combo2IsBomb = isBomb(combo2);
  
  if (combo1IsBomb && !combo2IsBomb) return true;
  if (!combo1IsBomb && combo2IsBomb) return false;
  
  if (combo1IsBomb && combo2IsBomb) {
    return compareBombs(combo1, combo2);
  }
  
  if (combo1.type !== combo2.type) return false;
  
  if (combo1.type === COMBINATION_TYPES.STRAIGHT || 
      combo1.type === COMBINATION_TYPES.SEQ_3_PAIRS || 
      combo1.type === COMBINATION_TYPES.SEQ_2_TRIPLES) {
    if (combo1.cards.length !== combo2.cards.length) return false;
  }
  
  return combo1.rank > combo2.rank;
};

export const detectBomb = (cards, currentRoundLevelRank = null) => {
  if (!cards || cards.length === 0) return null;
  if (!currentRoundLevelRank) return detectBombBase(cards, currentRoundLevelRank);

  const wildcards = cards.filter(c => c.suit === 'hearts' && c.rank === currentRoundLevelRank);
  const normalCards = cards.filter(c => !(c.suit === 'hearts' && c.rank === currentRoundLevelRank));

  if (wildcards.length === 0) return detectBombBase(cards, currentRoundLevelRank);

  let bestBomb = null;

  const evaluateSubstitutions = (currentNormal, remainingWildcards) => {
    if (remainingWildcards === 0) {
      const bomb = detectBombBase(currentNormal, currentRoundLevelRank);
      if (bomb) {
        const bombWithOriginalCards = { ...bomb, cards: cards };
        if (!bestBomb) {
          bestBomb = bombWithOriginalCards;
        } else {
          if (compareBombs(bombWithOriginalCards, bestBomb)) {
            bestBomb = bombWithOriginalCards;
          }
        }
      }
      return;
    }
    for (const subCard of ALL_STANDARD_CARDS) {
      evaluateSubstitutions([...currentNormal, subCard], remainingWildcards - 1);
    }
  };

  evaluateSubstitutions(normalCards, wildcards.length);
  return bestBomb;
};

const detectSingle = (cards, currentRoundLevelRank = null) => {
  if (cards.length !== 1) return null;
  const card = cards[0];
  
  if (card.joker) {
    return {
      type: COMBINATION_TYPES.SINGLE,
      rank: card.joker === 'red' ? 17 : 16,
      isValid: true,
      cards: cards
    };
  }
  
  return {
    type: COMBINATION_TYPES.SINGLE,
    rank: getRankValue(card.rank, currentRoundLevelRank),
    isValid: true,
    cards: cards
  };
};

const detectPair = (cards, currentRoundLevelRank = null) => {
  if (cards.length !== 2) return null;
  
  const redJokers = cards.filter(c => c.joker === 'red').length;
  const blackJokers = cards.filter(c => c.joker === 'black').length;
  
  if (redJokers === 2) {
    return {
      type: COMBINATION_TYPES.PAIR,
      rank: 17,
      isValid: true,
      cards: cards
    };
  }
  
  if (blackJokers === 2) {
    return {
      type: COMBINATION_TYPES.PAIR,
      rank: 16,
      isValid: true,
      cards: cards
    };
  }
  
  if (cards.some(c => c.joker)) return null;
  
  if (cards[0].rank === cards[1].rank) {
    return {
      type: COMBINATION_TYPES.PAIR,
      rank: getRankValue(cards[0].rank, currentRoundLevelRank),
      isValid: true,
      cards: cards
    };
  }
  
  return null;
};

const detectTriple = (cards, currentRoundLevelRank = null) => {
  if (cards.length !== 3) return null;
  if (cards.some(c => c.joker)) return null;
  
  if (cards[0].rank === cards[1].rank && cards[1].rank === cards[2].rank) {
    return {
      type: COMBINATION_TYPES.TRIPLE,
      rank: getRankValue(cards[0].rank, currentRoundLevelRank),
      isValid: true,
      cards: cards
    };
  }
  
  return null;
};

const detectStraight = (cards) => {
  if (cards.length !== 5) return null;
  if (cards.some(c => c.joker)) return null;
  
  const sorted = sortCardsByRank(cards, null);
  const ranks = sorted.map(c => c.rank);
  const uniqueRanks = [...new Set(ranks)];
  
  if (uniqueRanks.length !== cards.length) return null;
  
  const trySequence = (useAceAsOne) => {
    const values = sorted.map(c => getRankValueForStraight(c.rank, useAceAsOne));
    for (let i = 1; i < values.length; i++) {
      if (values[i] !== values[i - 1] + 1) return false;
    }
    if (useAceAsOne && values[values.length - 1] === 14) return false;
    if (!useAceAsOne && values[0] === 1) return false;
    return true;
  };
  
  const hasAce = ranks.includes('A');
  
  if (hasAce) {
    if (trySequence(true)) {
      return {
        type: COMBINATION_TYPES.STRAIGHT,
        rank: getRankValueForStraight(sorted[sorted.length - 1].rank, true),
        isValid: true,
        cards: cards
      };
    }
    if (trySequence(false)) {
      return {
        type: COMBINATION_TYPES.STRAIGHT,
        rank: getRankValueForStraight(sorted[sorted.length - 1].rank, false),
        isValid: true,
        cards: cards
      };
    }
    return null;
  }
  
  if (trySequence(false)) {
    return {
      type: COMBINATION_TYPES.STRAIGHT,
      rank: getRankValueForStraight(sorted[sorted.length - 1].rank, false),
      isValid: true,
      cards: cards
    };
  }
  
  return null;
};

const detectFullHouse = (cards, currentRoundLevelRank = null) => {
  if (cards.length !== 5) return null;
  if (cards.some(c => c.joker)) return null;
  
  const rankCounts = countRanks(cards);
  const counts = Object.values(rankCounts).sort((a, b) => b - a);
  
  if (counts.length !== 2 || counts[0] !== 3 || counts[1] !== 2) return null;
  
  const tripleRank = Object.keys(rankCounts).find(rank => rankCounts[rank] === 3);
  
  return {
    type: COMBINATION_TYPES.FULL_HOUSE,
    rank: getRankValue(tripleRank, currentRoundLevelRank),
    isValid: true,
    cards: cards
  };
};

const detectSeq3Pairs = (cards) => {
  if (cards.length !== 6) return null;
  if (cards.some(c => c.joker)) return null;
  
  const rankCounts = countRanks(cards);
  const pairs = Object.entries(rankCounts).filter(([_, count]) => count === 2);
  
  if (pairs.length !== 3) return null;
  
  const pairRanks = pairs.map(([rank, _]) => rank).sort((a, b) => getRankValueForStraight(a) - getRankValueForStraight(b));
  
  const trySequence = (useAceAsOne) => {
    const values = pairRanks.map(rank => getRankValueForStraight(rank, useAceAsOne));
    for (let i = 1; i < values.length; i++) {
      if (values[i] !== values[i - 1] + 1) return false;
    }
    if (useAceAsOne && values[values.length - 1] === 14) return false;
    if (!useAceAsOne && values[0] === 1) return false;
    return true;
  };
  
  const hasAce = pairRanks.includes('A');
  
  if (hasAce) {
    if (trySequence(true)) {
      return {
        type: COMBINATION_TYPES.SEQ_3_PAIRS,
        rank: getRankValueForStraight(pairRanks[pairRanks.length - 1], true),
        isValid: true,
        cards: cards
      };
    }
    if (trySequence(false)) {
      return {
        type: COMBINATION_TYPES.SEQ_3_PAIRS,
        rank: getRankValueForStraight(pairRanks[pairRanks.length - 1], false),
        isValid: true,
        cards: cards
      };
    }
    return null;
  }
  
  if (trySequence(false)) {
    return {
      type: COMBINATION_TYPES.SEQ_3_PAIRS,
      rank: getRankValueForStraight(pairRanks[pairRanks.length - 1], false),
      isValid: true,
      cards: cards
    };
  }
  
  return null;
};

const detectSeq2Triples = (cards) => {
  if (cards.length !== 6) return null;
  if (cards.some(c => c.joker)) return null;
  
  const rankCounts = countRanks(cards);
  const triples = Object.entries(rankCounts).filter(([_, count]) => count === 3);
  
  if (triples.length !== 2) return null;
  
  const tripleRanks = triples.map(([rank, _]) => rank).sort((a, b) => getRankValueForStraight(a) - getRankValueForStraight(b));
  
  const trySequence = (useAceAsOne) => {
    const values = tripleRanks.map(rank => getRankValueForStraight(rank, useAceAsOne));
    for (let i = 1; i < values.length; i++) {
      if (values[i] !== values[i - 1] + 1) return false;
    }
    if (useAceAsOne && values[values.length - 1] === 14) return false;
    if (!useAceAsOne && values[0] === 1) return false;
    return true;
  };
  
  const hasAce = tripleRanks.includes('A');
  
  if (hasAce) {
    if (trySequence(true)) {
      return {
        type: COMBINATION_TYPES.SEQ_2_TRIPLES,
        rank: getRankValueForStraight(tripleRanks[tripleRanks.length - 1], true),
        isValid: true,
        cards: cards
      };
    }
    if (trySequence(false)) {
      return {
        type: COMBINATION_TYPES.SEQ_2_TRIPLES,
        rank: getRankValueForStraight(tripleRanks[tripleRanks.length - 1], false),
        isValid: true,
        cards: cards
      };
    }
    return null;
  }
  
  if (trySequence(false)) {
    return {
      type: COMBINATION_TYPES.SEQ_2_TRIPLES,
      rank: getRankValueForStraight(tripleRanks[tripleRanks.length - 1], false),
      isValid: true,
      cards: cards
    };
  }
  
  return null;
};

const detectCombinationBase = (cards, currentRoundLevelRank = null) => {
  if (!cards || cards.length === 0) {
    return { type: COMBINATION_TYPES.INVALID, rank: 0, isValid: false };
  }
  
  const bomb = detectBombBase(cards, currentRoundLevelRank);
  if (bomb) return bomb;
  
  const detectors = [
    (c) => detectSingle(c, currentRoundLevelRank),
    (c) => detectPair(c, currentRoundLevelRank),
    (c) => detectTriple(c, currentRoundLevelRank),
    (c) => detectFullHouse(c, currentRoundLevelRank),
    (c) => detectSeq3Pairs(c),
    (c) => detectSeq2Triples(c),
    (c) => detectStraight(c)
  ];
  
  for (const detector of detectors) {
    const result = detector(cards);
    if (result) return result;
  }
  
  return { type: COMBINATION_TYPES.INVALID, rank: 0, isValid: false };
};

export const detectCombination = (cards, currentRoundLevelRank = null) => {
  if (!cards || cards.length === 0) {
    return { type: COMBINATION_TYPES.INVALID, rank: 0, isValid: false };
  }

  if (!currentRoundLevelRank) {
    return detectCombinationBase(cards, currentRoundLevelRank);
  }

  const wildcards = cards.filter(c => c.suit === 'hearts' && c.rank === currentRoundLevelRank);
  const normalCards = cards.filter(c => !(c.suit === 'hearts' && c.rank === currentRoundLevelRank));

  if (wildcards.length === 0) {
    return detectCombinationBase(cards, currentRoundLevelRank);
  }

  let bestCombo = { type: COMBINATION_TYPES.INVALID, rank: 0, isValid: false, bombStrength: 0 };

  const evaluateSubstitutions = (currentNormal, remainingWildcards) => {
    if (remainingWildcards === 0) {
      const combo = detectCombinationBase(currentNormal, currentRoundLevelRank);
      if (combo.isValid) {
        const comboWithOriginalCards = { ...combo, cards: cards };
        if (!bestCombo.isValid) {
          bestCombo = comboWithOriginalCards;
        } else {
          if (compareCombinations(comboWithOriginalCards, bestCombo, currentRoundLevelRank)) {
            bestCombo = comboWithOriginalCards;
          }
        }
      }
      return;
    }

    for (const subCard of ALL_STANDARD_CARDS) {
      evaluateSubstitutions([...currentNormal, subCard], remainingWildcards - 1);
    }
  };

  evaluateSubstitutions(normalCards, wildcards.length);

  return bestCombo;
};

export const validatePlay = (selectedCards, lastPlay, currentRoundLevelRank = null) => {
  const currentCombo = detectCombination(selectedCards, currentRoundLevelRank);
  
  if (!currentCombo.isValid) {
    return {
      isValid: false,
      error: 'Selected cards do not form a valid combination'
    };
  }
  
  if (!lastPlay || lastPlay.length === 0) {
    return { isValid: true, error: null };
  }
  
  const lastCombo = detectCombination(lastPlay, currentRoundLevelRank);
  
  if (!lastCombo.isValid) {
    return { isValid: true, error: null };
  }
  
  const currentIsBomb = isBomb(currentCombo);
  const lastIsBomb = isBomb(lastCombo);
  
  if (lastIsBomb && !currentIsBomb) {
    return {
      isValid: false,
      error: 'Cannot play normal combination after bomb - must play higher bomb or pass'
    };
  }
  
  if (currentIsBomb && !lastIsBomb) {
    return { isValid: true, error: null };
  }
  
  if (currentIsBomb && lastIsBomb) {
    if (!compareBombs(currentCombo, lastCombo)) {
      return {
        isValid: false,
        error: 'Your bomb is not stronger than the last bomb'
      };
    }
    return { isValid: true, error: null };
  }
  
  if (currentCombo.type !== lastCombo.type) {
    return {
      isValid: false,
      error: `Must play same combination type: ${lastCombo.type.toLowerCase().replace(/_/g, ' ')}`
    };
  }
  
  if (currentCombo.type === COMBINATION_TYPES.STRAIGHT || 
      currentCombo.type === COMBINATION_TYPES.SEQ_3_PAIRS || 
      currentCombo.type === COMBINATION_TYPES.SEQ_2_TRIPLES) {
    if (currentCombo.cards.length !== lastCombo.cards.length) {
      return {
        isValid: false,
        error: `Must play same length: ${lastCombo.cards.length} cards`
      };
    }
  }
  
  if (!compareCombinations(currentCombo, lastCombo, currentRoundLevelRank)) {
    return {
      isValid: false,
      error: 'Your combination is not stronger than the last play'
    };
  }
  
  return { isValid: true, error: null };
};

export { COMBINATION_TYPES, BOMB_TYPES };