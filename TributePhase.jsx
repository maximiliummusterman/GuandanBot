import React, { useState, useEffect, useRef } from 'react';
import pb from '@/lib/pocketbaseClient.js';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import CardDisplay from '@/components/CardDisplay.jsx';
import { getHighestCard } from '@/utils/getHighestCard.js';
import { isValidReturnCard } from '@/utils/isValidReturnCard.js';
import { cn } from '@/lib/utils.js';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
// Tunnel: tribute-phase match updates also write the durable bot ledger.
import { buildTunnelMatchUpdate } from './matchTunnel.jsx';

const TributePhase = ({ currentMatch, currentPlayer, allMatchPlayers, currentUser, onCardSelect, selectedCards, tributeRef }) => {
  const [tributeState, setTributeState] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPhaseCompleting, setIsPhaseCompleting] = useState(false);
  const navigate = useNavigate();

  // ==========================================
  // UI-FREEZE MECHANISMUS
  // ==========================================
  const cachedMatch = useRef(currentMatch);
  const cachedPlayers = useRef(allMatchPlayers);
  const cachedCurrentPlayer = useRef(currentPlayer);

  const isEffectivelyCompleting = isPhaseCompleting || currentMatch?.gameStatus === 'playing';

  if (!isEffectivelyCompleting) {
    cachedMatch.current = currentMatch;
    cachedPlayers.current = allMatchPlayers;
    cachedCurrentPlayer.current = currentPlayer;
  }

  const safeMatch = isEffectivelyCompleting ? cachedMatch.current : currentMatch;
  const safePlayers = isEffectivelyCompleting ? cachedPlayers.current : allMatchPlayers;
  const safeCurrentPlayer = isEffectivelyCompleting ? cachedCurrentPlayer.current : currentPlayer;
  // ==========================================

  const roundLevelRank = safeMatch.currentRoundLevelRank === 'Blue'
    ? safeMatch.levelRankBlue
    : safeMatch.levelRankRed;

  const trickCards = safeMatch.trickLastPlay || [];

  // Normal State Tracking
  const tributeCard = trickCards[0];
  const returnCard = trickCards[1];
  const exchangeState = trickCards.length === 0 ? 1 : trickCards.length === 1 ? 2 : 3;

  // EXCEPTION_1 State Tracking (Special Case 1)
  const isException1 = tributeState?.type === 'EXCEPTION_1';
  const tributesE1 = isException1 ? trickCards.filter(c => c._type === 'tribute') : [];
  const returnsE1 = isException1 ? trickCards.filter(c => c._type === 'return') : [];

  let requiredPlaceE1 = null;
  if (isException1) {
    const tLen = tributesE1.length;
    const rLen = returnsE1.length;

    if (tLen === 2 && rLen === 2) requiredPlaceE1 = 'take_returns';
    else if (tLen === 1 && rLen === 1) requiredPlaceE1 = 'take_returns';
    else if (tLen === 2 && rLen === 1) requiredPlaceE1 = '2';
    else if (tLen === 2 && rLen === 0) requiredPlaceE1 = '1';
    else if (tLen === 1 && rLen === 0) requiredPlaceE1 = '4';
    else if (tLen === 0 && rLen === 0) requiredPlaceE1 = '3';
    else requiredPlaceE1 = 'take_returns'; // fallback
  }

  const getCardValue = (card, levelRank) => {
    if (card.joker === 'red') return 1000;
    if (card.joker === 'black') return 900;
    if (card.rank === levelRank && card.suit === 'hearts') return 800;
    if (card.rank === levelRank) return 700;
    const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
    return Math.max(0, ranks.indexOf(card.rank));
  };

  const assignedTributes = () => {
    if (tributesE1.length !== 2) return null;
    const val0 = getCardValue(tributesE1[0], roundLevelRank);
    const val1 = getCardValue(tributesE1[1], roundLevelRank);

    // Higher card goes to 1st place, Lower card goes to 2nd place
    const higher = val0 >= val1 ? tributesE1[0] : tributesE1[1];
    const lower = val0 >= val1 ? tributesE1[1] : tributesE1[0];

    return {
      '1': higher,
      '2': lower
    };
  };

  const detectException = () => {
    const finishedPlayers = safePlayers
      .filter(p => p.finishPlace)
      .sort((a, b) => parseInt(a.finishPlace) - parseInt(b.finishPlace));

    const firstPlace = finishedPlayers.find(p => p.finishPlace === '1');
    const secondPlace = finishedPlayers.find(p => p.finishPlace === '2');
    const thirdPlace = finishedPlayers.find(p => p.finishPlace === '3');
    const fourthPlace = finishedPlayers.find(p => p.finishPlace === '4');

    const countRedJokers = (hand) => {
      if (!hand) return 0;
      return hand.filter(card => card.joker === 'red').length;
    };

    // 1. Check Anti-Tribute FIRST (EXCEPTION_2a and 2b)
    if (fourthPlace) {
      const fourthPlaceRedJokers = countRedJokers(fourthPlace.hand);
      if (fourthPlaceRedJokers === 2) {
        return {
          type: 'EXCEPTION_2a',
          loser: fourthPlace,
          description: 'No tribute - loser has both red jokers'
        };
      }
    }

    if (thirdPlace && fourthPlace && thirdPlace.team === fourthPlace.team) {
      const thirdPlaceRedJokers = countRedJokers(thirdPlace.hand);
      const fourthPlaceRedJokers = countRedJokers(fourthPlace.hand);
      const totalRedJokers = thirdPlaceRedJokers + fourthPlaceRedJokers;

      if (totalRedJokers > 0) {
        if (fourthPlaceRedJokers === 2 || thirdPlaceRedJokers === 2) {
          const defender = fourthPlaceRedJokers === 2 ? fourthPlace : thirdPlace;
          return {
            type: 'EXCEPTION_2b_CASE_1',
            loserWithJokers: defender,
            description: 'Tribute completely defended - one player has both red jokers'
          };
        }
        if (thirdPlaceRedJokers === 1 && fourthPlaceRedJokers === 1) {
          return {
            type: 'EXCEPTION_2b_CASE_2',
            losers: [thirdPlace, fourthPlace],
            description: 'Both losers show their red joker, no tribute'
          };
        }
      }
    }

    // 2. Check Doppel-Tribute (EXCEPTION_1) ONLY if Anti-Tribute is not possible
    if (firstPlace && secondPlace && firstPlace.team === secondPlace.team) {
      const losers = [thirdPlace, fourthPlace].filter(Boolean);
      const winners = [firstPlace, secondPlace];
      return {
        type: 'EXCEPTION_1',
        losers: losers,
        winners: winners,
        description: 'Both losers give highest card to both winners'
      };
    }

    // 3. Basic Tribute
    return {
      type: 'BASIC',
      losers: fourthPlace ? [fourthPlace] : [],
      winners: firstPlace ? [firstPlace] : [],
      description: 'Standard tribute: finishPlace 4 gives highest card to finishPlace 1'
    };
  };

  const isCurrentPlayerLoser = () => {
    if (!tributeState) return false;
    if (tributeState.type === 'BASIC' || tributeState.type === 'EXCEPTION_1') {
      return tributeState.losers.some(l => l.user_id === currentUser.id);
    }
    return false;
  };

  const isCurrentPlayerWinner = () => {
    if (!tributeState) return false;
    if (tributeState.type === 'BASIC') {
      return safeCurrentPlayer.finishPlace === '1';
    }
    if (tributeState.type === 'EXCEPTION_1') {
      return tributeState.winners.some(w => w.user_id === currentUser.id);
    }
    return false;
  };

  const getTributeReceiver = () => {
    if (!tributeState) return null;
    if (tributeState.type === 'BASIC') {
      return safePlayers.find(p => p.finishPlace === '1');
    }
    if (tributeState.type === 'EXCEPTION_1') {
      if (safeCurrentPlayer.finishPlace === '3') {
        return safePlayers.find(p => p.finishPlace === '1');
      } else if (safeCurrentPlayer.finishPlace === '4') {
        return safePlayers.find(p => p.finishPlace === '2');
      }
    }
    return null;
  };

  const validateTributeCard = (card) => {
    const highestCard = getHighestCard(safeCurrentPlayer.hand, roundLevelRank);
    if (!highestCard) return { isValid: false, error: 'No valid cards to give' };

    const isSameCard = card.joker
      ? card.joker === highestCard.joker
      : card.rank === highestCard.rank;

    if (!isSameCard) {
      return { isValid: false, error: 'You must select your highest card' };
    }
    return { isValid: true };
  };

  useEffect(() => {
    if (isEffectivelyCompleting) return;
    const exceptionCase = detectException();
    setTributeState(exceptionCase);
  }, [safePlayers, isEffectivelyCompleting]);

  const handlePlayTribute = async () => {
    if (!selectedCards || selectedCards.length === 0 || isSubmitting || isEffectivelyCompleting) return;

    const selectedCard = selectedCards[0];
    const validation = validateTributeCard(selectedCard);

    if (!validation.isValid) {
      toast.error(validation.error);
      return;
    }

    setIsSubmitting(true);
    try {
      const updatedHand = safeCurrentPlayer.hand.filter((handCard, index) => {
        const isMatch = handCard.joker && selectedCard.joker
          ? handCard.joker === selectedCard.joker
          : handCard.suit === selectedCard.suit && handCard.rank === selectedCard.rank;

        if (isMatch) {
          const alreadyRemoved = safeCurrentPlayer.hand.slice(0, index).some(prevCard =>
            prevCard.joker && selectedCard.joker
              ? prevCard.joker === selectedCard.joker
              : prevCard.suit === selectedCard.suit && prevCard.rank === selectedCard.rank
          );
          return alreadyRemoved;
        }
        return true;
      });

      await pb.collection('matchplayers').update(safeCurrentPlayer.id, {
        hand: updatedHand
      }, { $autoCancel: false });

      const latestMatch = await pb.collection('matches').getOne(safeMatch.id, { $autoCancel: false });
      const latestTrick = latestMatch.trickLastPlay || [];

      let newTrick = [];
      if (isException1) {
        newTrick = [...latestTrick, {
          ...selectedCard,
          _type: 'tribute',
          _giverId: currentUser.id,
          _giverPlace: safeCurrentPlayer.finishPlace,
          _giverUsername: safeCurrentPlayer.expand?.user_id?.username
        }];
      } else {
        newTrick = [selectedCard];
      }

      // Tunnel: tribute plays are stored as explicit events because PocketBase only keeps one visible trick array.
      const matchUpdateData = buildTunnelMatchUpdate(
        latestMatch,
        safePlayers,
        {
          trickLastPlay: newTrick
        },
        {
          type: 'tribute_play',
          seat: safeCurrentPlayer.seat,
          cards: [selectedCard],
          tributeType: tributeState?.type || 'BASIC'
        }
      );

      await pb.collection('matches').update(safeMatch.id, matchUpdateData, { $autoCancel: false });

      toast.success('Tribute card played');
      if (onCardSelect) onCardSelect([]);

    } catch (err) {
      toast.error('Failed to play tribute card');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePlayReturn = async () => {
    if (!selectedCards || selectedCards.length === 0 || isSubmitting || isEffectivelyCompleting) return;

    const selectedCard = selectedCards[0];
    if (!isValidReturnCard(selectedCard, roundLevelRank)) {
      toast.error('Return card must be ≤10 and not the level rank');
      return;
    }

    setIsSubmitting(true);
    try {
      const currentHand = safeCurrentPlayer.hand || [];
      const updatedHand = currentHand.filter((handCard, index) => {
        const isMatch = handCard.joker && selectedCard.joker
          ? handCard.joker === selectedCard.joker
          : handCard.suit === selectedCard.suit && handCard.rank === selectedCard.rank;

        if (isMatch) {
          const alreadyRemoved = currentHand.slice(0, index).some(prevCard =>
            prevCard.joker && selectedCard.joker
              ? prevCard.joker === selectedCard.joker
              : prevCard.suit === selectedCard.suit && prevCard.rank === selectedCard.rank
          );
          return alreadyRemoved;
        }
        return true;
      });

      const latestMatch = await pb.collection('matches').getOne(safeMatch.id, { $autoCancel: false });
      const latestTrick = latestMatch.trickLastPlay || [];

      let newTrick = [];
      let matchUpdateData = {};
      
      if (isException1) {
        const assignments = assignedTributes();
        if (!assignments) {
          setIsSubmitting(false);
          return;
        }

        const assigned = assignments[safeCurrentPlayer.finishPlace];
        const cleanTribute = { suit: assigned.suit, rank: assigned.rank, joker: assigned.joker };
        updatedHand.push(cleanTribute);

        newTrick = [...latestTrick, {
          ...selectedCard,
          _type: 'return',
          _returnerId: currentUser.id,
          _targetGiverId: assigned._giverId,
          _returnerUsername: safeCurrentPlayer.expand?.user_id?.username
        }];

        // Exception 1 startSeat: The player who tributed the higher card to 1st place starts
        if (safeCurrentPlayer.finishPlace === '1') {
          const targetPlayer = safePlayers.find(p => p.user_id === assignments['1']._giverId);
          if (targetPlayer) {
            matchUpdateData.startSeat = parseInt(targetPlayer.seat);
          }
        }
      } else {
        newTrick = [tributeCard, selectedCard];
      }

      await pb.collection('matchplayers').update(safeCurrentPlayer.id, {
        hand: updatedHand
      }, { $autoCancel: false });

      matchUpdateData.trickLastPlay = newTrick;

      // Tunnel: return selections get their own ledger event so the Python side can replay the whole tribute exchange.
      const tunnelMatchUpdate = buildTunnelMatchUpdate(
        latestMatch,
        safePlayers,
        matchUpdateData,
        {
          type: 'return_play',
          seat: safeCurrentPlayer.seat,
          cards: [selectedCard],
          tributeType: tributeState?.type || 'BASIC'
        }
      );

      await pb.collection('matches').update(safeMatch.id, tunnelMatchUpdate, { $autoCancel: false });

      toast.success('Return card played');
      if (onCardSelect) onCardSelect([]);

    } catch (err) {
      toast.error('Failed to play return card');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTakeReturn = async (clickedReturnCard) => {
    if (isSubmitting || isEffectivelyCompleting) return;
    setIsSubmitting(true);

    try {
      if (isException1) {
        if (clickedReturnCard._targetGiverId !== currentUser.id) {
          toast.error('This return card is not for you!');
          setIsSubmitting(false);
          return;
        }

        const cleanReturnCard = {
          suit: clickedReturnCard.suit,
          rank: clickedReturnCard.rank,
          joker: clickedReturnCard.joker
        };

        const updatedHand = [...(safeCurrentPlayer.hand || []), cleanReturnCard];
        await pb.collection('matchplayers').update(safeCurrentPlayer.id, {
          hand: updatedHand
        }, { $autoCancel: false });

        const latestMatch = await pb.collection('matches').getOne(safeMatch.id, { $autoCancel: false });
        const currentTrick = latestMatch.trickLastPlay || [];

        const updatedTrickCards = currentTrick.filter(c => {
          const isMyReturn = c._type === 'return' && c._targetGiverId === currentUser.id;
          const isMyTribute = c._type === 'tribute' && c._giverId === currentUser.id;
          return !isMyReturn && !isMyTribute;
        });

        if (updatedTrickCards.length === 0) {
          setIsPhaseCompleting(true);

          // Retrieve the startSeat previously calculated and stored by 1st place in handlePlayReturn
          const finalStartSeat = latestMatch.startSeat || 1;

          // Tunnel: closing tribute and starting play clears the tunnel trick state for the new live trick cycle.
          const matchUpdateData = buildTunnelMatchUpdate(
            latestMatch,
            safePlayers,
            {
              trickLastPlay: [],
              gameStatus: 'playing',
              currentSeat: finalStartSeat.toString(),
              startSeat: finalStartSeat
            },
            {
              type: 'phase_reset',
              phase: 'playing',
              currentSeat: finalStartSeat,
              clearCurrentTrick: true,
              reason: 'exception_1_complete'
            }
          );

          await pb.collection('matches').update(safeMatch.id, matchUpdateData, { $autoCancel: false });

          const allPlayers = await pb.collection('matchplayers').getFullList({
            filter: `match_id = "${safeMatch.id}"`,
            $autoCancel: false
          });
          for (const player of allPlayers) {
            await pb.collection('matchplayers').update(player.id, { finishPlace: null }, { $autoCancel: false });
          }

          toast.success('Tribute phase complete! Game starting.');
          navigate(`/match/${safeMatch.id}`);
        } else {
          // Tunnel: partial exception-1 takebacks still need a ledger event even though the visible tribute array shrinks.
          const matchUpdateData = buildTunnelMatchUpdate(
            latestMatch,
            safePlayers,
            {
              trickLastPlay: updatedTrickCards
            },
            {
              type: 'return_taken',
              seat: safeCurrentPlayer.seat,
              cards: [clickedReturnCard],
              tributeType: tributeState?.type || 'EXCEPTION_1'
            }
          );

          await pb.collection('matches').update(safeMatch.id, matchUpdateData, { $autoCancel: false });
          toast.success('Return card taken.');
        }

      } else {
        setIsPhaseCompleting(true);

        const cleanReturn = { suit: returnCard.suit, rank: returnCard.rank, joker: returnCard.joker };
        const updatedTributerHand = [...(safeCurrentPlayer.hand || []), cleanReturn];
        await pb.collection('matchplayers').update(safeCurrentPlayer.id, {
          hand: updatedTributerHand
        }, { $autoCancel: false });

        const receiver = getTributeReceiver();
        if (receiver) {
          const cleanTribute = { suit: tributeCard.suit, rank: tributeCard.rank, joker: tributeCard.joker };
          const updatedReceiverHand = [...(receiver.hand || []), cleanTribute];
          await pb.collection('matchplayers').update(receiver.id, {
            hand: updatedReceiverHand
          }, { $autoCancel: false });
        }

        // Normal Tribute startSeat: finishPlace 4
        let nextStartSeat = 1;
        const fourthPlace = safePlayers.find(p => p.finishPlace === '4');
        if (fourthPlace) {
          nextStartSeat = parseInt(fourthPlace.seat);
        }

        // Tunnel: basic tribute completion resets the tunnel trick state before regular play begins.
        const matchUpdateData = buildTunnelMatchUpdate(
          safeMatch,
          safePlayers,
          {
            trickLastPlay: [],
            gameStatus: 'playing',
            currentSeat: nextStartSeat.toString(),
            startSeat: nextStartSeat
          },
          {
            type: 'phase_reset',
            phase: 'playing',
            currentSeat: nextStartSeat,
            clearCurrentTrick: true,
            reason: 'basic_tribute_complete'
          }
        );

        await pb.collection('matches').update(safeMatch.id, matchUpdateData, { $autoCancel: false });

        const allPlayers = await pb.collection('matchplayers').getFullList({
          filter: `match_id = "${safeMatch.id}"`,
          $autoCancel: false
        });

        for (const player of allPlayers) {
          await pb.collection('matchplayers').update(player.id, {
            finishPlace: null
          }, { $autoCancel: false });
        }

        toast.success('Tribute phase complete! Game starting.');
        navigate(`/match/${safeMatch.id}`);
      }

    } catch (err) {
      toast.error('Failed to complete tribute exchange');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSkipTribute = async () => {
    if (isSubmitting || isEffectivelyCompleting) return;
    setIsSubmitting(true);
    setIsPhaseCompleting(true);

    try {
      // antiTribute (normal tribute defended): finishPlace 1 gets startSeat
      let startSeat = 1;
      const firstPlacePlayer = safePlayers.find(p => p.finishPlace === '1');
      if (firstPlacePlayer) {
        startSeat = parseInt(firstPlacePlayer.seat);
      }

      // Tunnel: anti-tribute still needs a phase-reset event so the bot tunnel knows tribute ended without card exchange.
      const matchUpdateData = buildTunnelMatchUpdate(
        safeMatch,
        safePlayers,
        {
          gameStatus: 'playing',
          currentSeat: startSeat.toString(),
          startSeat: startSeat
        },
        {
          type: 'phase_reset',
          phase: 'playing',
          currentSeat: startSeat,
          clearCurrentTrick: true,
          reason: 'tribute_skipped'
        }
      );

      await pb.collection('matches').update(safeMatch.id, matchUpdateData, { $autoCancel: false });

      const allPlayers = await pb.collection('matchplayers').getFullList({
        filter: `match_id = "${safeMatch.id}"`,
        $autoCancel: false
      });

      for (const player of allPlayers) {
        await pb.collection('matchplayers').update(player.id, {
          finishPlace: null
        }, { $autoCancel: false });
      }

      toast.success('Tribute phase skipped - starting game');
      navigate(`/match/${safeMatch.id}`);
    } catch (err) {
      toast.error('Failed to skip tribute phase');
    } finally {
      setIsSubmitting(false);
    }
  };

  const canPlayTribute = () => {
    if (isEffectivelyCompleting || isSubmitting) return false;
    if (!selectedCards?.length || !validateTributeCard(selectedCards[0]).isValid) return false;
    if (isException1) return safeCurrentPlayer.finishPlace === requiredPlaceE1;
    return isCurrentPlayerLoser() && exchangeState === 1;
  };

  const canPlayReturn = () => {
    if (isEffectivelyCompleting || isSubmitting) return false;
    if (!selectedCards?.length || !isValidReturnCard(selectedCards[0], roundLevelRank)) return false;
    if (isException1) return safeCurrentPlayer.finishPlace === requiredPlaceE1;
    return isCurrentPlayerWinner() && exchangeState === 2;
  };

  if (tributeRef) {
    tributeRef.current = {
      exchangeState,
      requiredPlaceE1,
      isException1,
      canPlayTribute: canPlayTribute(),
      canPlayReturn: canPlayReturn(),
      handlePlayTribute,
      handlePlayReturn,
      isSubmitting,
      isCurrentPlayerLoser: isCurrentPlayerLoser(),
      isCurrentPlayerWinner: isCurrentPlayerWinner(),
      currentPlayerPlace: safeCurrentPlayer.finishPlace,
      isEffectivelyCompleting
    };
  }

  if (isEffectivelyCompleting) {
    return (
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-md px-4 z-10">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-card/95 backdrop-blur-md rounded-2xl p-8 shadow-2xl border border-border flex flex-col items-center justify-center gap-6"
        >
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          <div className="text-center space-y-2">
            <h3 className="text-xl font-bold text-foreground">Completing tribute phase...</h3>
            <p className="text-sm text-muted-foreground">Preparing the game board</p>
          </div>
        </motion.div>
      </div>
    );
  }

  if (!tributeState) return null;

  if (['EXCEPTION_2a', 'EXCEPTION_2b_CASE_1', 'EXCEPTION_2b_CASE_2'].includes(tributeState.type)) {
    return (
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl px-4 z-10">
        <div className="bg-card/95 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-border">
          <div className="text-center space-y-4">
            <h3 className="text-xl font-bold">Tribute Phase - Abgewehrt</h3>
            <p className="text-sm text-muted-foreground">{tributeState.description}</p>

            {tributeState.type === 'EXCEPTION_2a' && (
              <div className="space-y-4">
                <p className="text-lg font-medium">
                  {tributeState.loser.expand?.user_id?.username} hat beide roten Joker
                </p>
                <div className="flex gap-2 justify-center">
                  <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                  <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                </div>
              </div>
            )}

            {tributeState.type === 'EXCEPTION_2b_CASE_1' && (
              <div className="space-y-4">
                <p className="text-lg font-medium">
                  {tributeState.loserWithJokers.expand?.user_id?.username} hat beide roten Joker
                </p>
                <div className="flex gap-2 justify-center">
                  <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                  <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                </div>
              </div>
            )}

            {tributeState.type === 'EXCEPTION_2b_CASE_2' && (
              <div className="space-y-4">
                <p className="text-lg font-medium">Beide Spieler im Verlierer-Team haben je einen roten Joker</p>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium mb-2">{tributeState.losers[0].expand?.user_id?.username}</p>
                    <div className="flex gap-2 justify-center">
                      <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-2">{tributeState.losers[1].expand?.user_id?.username}</p>
                    <div className="flex gap-2 justify-center">
                      <CardDisplay card={{ joker: 'red' }} levelRank={roundLevelRank} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <Button
              onClick={handleSkipTribute}
              disabled={isSubmitting || isEffectivelyCompleting}
              className="w-full mt-4"
            >
              {isSubmitting || isEffectivelyCompleting ? 'Starting game...' : 'Start Game'}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const getStatusMessage = () => {
    if (isException1) {
      if (requiredPlaceE1 === '3') {
        return safeCurrentPlayer.finishPlace === '3'
          ? 'Select your highest card to give as tribute (1st of 2)'
          : 'Waiting for 3rd place to play tribute...';
      }
      if (requiredPlaceE1 === '4') {
        return safeCurrentPlayer.finishPlace === '4'
          ? 'Select your highest card to give as tribute (2nd of 2)'
          : 'Waiting for 4th place to play tribute...';
      }
      if (requiredPlaceE1 === '1') {
        return safeCurrentPlayer.finishPlace === '1'
          ? 'Select a return card for your tribute'
          : 'Waiting for 1st place to return a card...';
      }
      if (requiredPlaceE1 === '2') {
        return safeCurrentPlayer.finishPlace === '2'
          ? 'Select a return card for your tribute'
          : 'Waiting for 2nd place to return a card...';
      }
      if (requiredPlaceE1 === 'take_returns') {
        const myReturn = returnsE1.find(c => c._targetGiverId === currentUser.id);
        if (isCurrentPlayerLoser() && myReturn) {
          return 'Click your return card to take it and start the game';
        }
        return 'Waiting for tributers to take their return cards...';
      }
    } else {
      if (exchangeState === 1) {
        return isCurrentPlayerLoser()
          ? 'Select your highest card to give as tribute'
          : 'Waiting for tributer to play a card...';
      }
      if (exchangeState === 2) {
        return isCurrentPlayerWinner()
          ? 'Select a return card (≤10, not level rank)'
          : 'Waiting for receiver to return a card...';
      }
      if (exchangeState === 3) {
        return isCurrentPlayerLoser()
          ? 'Click the return card to take it and start the game'
          : 'Waiting for tributer to take the return card...';
      }
    }
    return 'Tribute phase in progress';
  };

  return (
    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-4xl px-4 flex flex-col items-center gap-12 z-10">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-background/80 backdrop-blur-md border border-border shadow-lg px-8 py-3 rounded-full"
      >
        <p className="text-lg font-medium text-foreground text-center">
          {getStatusMessage()}
        </p>
      </motion.div>

      <div className="flex flex-col items-center justify-center gap-8 min-h-[200px] w-full">
        {isException1 ? (
          <>
            {tributesE1.length > 0 ? (
              <div className="flex items-center justify-center gap-12">
                {tributesE1.map((t, idx) => (
                  <motion.div
                    key={`tribute-${idx}`}
                    initial={{ opacity: 0, scale: 0.8, y: -20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    className="flex flex-col items-center gap-4"
                  >
                    <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest">
                      Tribute ({t._giverUsername})
                    </span>
                    <CardDisplay card={t} levelRank={roundLevelRank} />
                  </motion.div>
                ))}
              </div>
            ) : null}

            {returnsE1.length > 0 ? (
              <div className="flex items-center justify-center gap-12">
                {returnsE1.map((r, idx) => {
                  const isMyReturn = r._targetGiverId === currentUser.id;
                  const canTake = isCurrentPlayerLoser() && requiredPlaceE1 === 'take_returns' && isMyReturn && !isEffectivelyCompleting;
                  return (
                    <motion.div
                      key={`return-${idx}`}
                      initial={{ opacity: 0, scale: 0.8, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      className="flex flex-col items-center gap-4"
                    >
                      <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest">
                        Return ({r._returnerUsername})
                      </span>
                      <div
                        onClick={canTake ? () => handleTakeReturn(r) : undefined}
                        className={cn(
                          "transition-all duration-300",
                          canTake
                            ? "cursor-pointer hover:-translate-y-4 hover:shadow-2xl ring-4 ring-primary/50 rounded-xl"
                            : ""
                        )}
                      >
                        <CardDisplay card={r} levelRank={roundLevelRank} />
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            ) : null}
          </>
        ) : (
          <div className="flex items-center justify-center gap-12">
            {tributeCard && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8, x: -20 }}
                animate={{ opacity: 1, scale: 1, x: 0 }}
                className="flex flex-col items-center gap-4"
              >
                <span className="text-sm font-bold text-muted-foreground uppercase tracking-widest">Tribute Card</span>
                <CardDisplay card={tributeCard} levelRank={roundLevelRank} />
              </motion.div>
            )}

            {returnCard && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8, x: 20 }}
                animate={{ opacity: 1, scale: 1, x: 0 }}
                className="flex flex-col items-center gap-4"
              >
                <span className="text-sm font-bold text-muted-foreground uppercase tracking-widest">Return Card</span>
                <div
                  onClick={isCurrentPlayerLoser() && exchangeState === 3 && !isEffectivelyCompleting ? () => handleTakeReturn(returnCard) : undefined}
                  className={cn(
                    "transition-all duration-300",
                    isCurrentPlayerLoser() && exchangeState === 3 && !isEffectivelyCompleting
                      ? "cursor-pointer hover:-translate-y-4 hover:shadow-2xl ring-4 ring-primary/50 rounded-xl"
                      : ""
                  )}
                >
                  <CardDisplay card={returnCard} levelRank={roundLevelRank} />
                </div>
              </motion.div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TributePhase;
