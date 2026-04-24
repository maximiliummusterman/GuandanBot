export const isValidReturnCard = (card, levelRank) => {
  if (!card) return false;
  
  if (card.joker) return false;
  
  if (card.rank === levelRank) return false;
  
  const validRanks = ['2', '3', '4', '5', '6', '7', '8', '9', '10'];
  
  return validRanks.includes(card.rank);
};