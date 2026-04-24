const getRankValue = (rank) => {
  const rankOrder = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
  };
  
  return rankOrder[rank] || 0;
};

export const getHighestCard = (hand, roundLevelRank) => {
  if (!hand || hand.length === 0) return null;
  
  const filteredHand = hand.filter(card => {
    if (card.joker) return true;
    
    if (card.suit === 'hearts' && card.rank === roundLevelRank) {
      return false;
    }
    
    return true;
  });
  
  if (filteredHand.length === 0) return null;
  
  let highestCard = filteredHand[0];
  let highestValue = -1;
  
  for (const card of filteredHand) {
    let cardValue;
    
    if (card.joker === 'red') {
      cardValue = 1000;
    } else if (card.joker === 'black') {
      cardValue = 999;
    } else if (card.rank === roundLevelRank) {
      cardValue = 100;
    } else {
      cardValue = getRankValue(card.rank);
    }
    
    if (cardValue > highestValue) {
      highestValue = cardValue;
      highestCard = card;
    }
  }
  
  return highestCard;
};