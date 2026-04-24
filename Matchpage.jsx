import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext.jsx';
import pb from '@/lib/pocketbaseClient.js';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Helmet } from 'react-helmet';
import { RefreshCw, Play, SkipForward, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import DeckCard from '@/components/DeckCard.jsx';
import CurrentPlayerHand from '@/components/CurrentPlayerHand.jsx';
import TrickDisplay from '@/components/TrickDisplay.jsx';
import TributePhase from '@/components/TributePhase.jsx';
import CardDisplay from '@/components/CardDisplay.jsx';
import { validatePlay, detectCombination, isBomb } from '@/utils/cardCombinations.js';
// Tunnel: every match mutation also updates the durable PocketBase bot ledger.
import { buildTunnelMatchUpdate } from './matchTunnel.jsx';

const MatchPage = () => {
  const { matchId } = useParams();
  const navigate = useNavigate();
  const { currentUser } = useAuth();

  const [match, setMatch] = useState(null);
  const [matchPlayers, setMatchPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const [deckLocked, setDeckLocked] = useState(false);
  const [selectedCards, setSelectedCards] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gameWinner, setGameWinner] = useState(null);
  const [drawnCard, setDrawnCard] = useState(null);

  const drawTimeoutRef = useRef(null);
  const tributeRef = useRef({});
  const matchPlayersRef = useRef([]);

  useEffect(() => {
    matchPlayersRef.current = matchPlayers;
  }, [matchPlayers]);

  const createDeck = () => {
    const suits = ['hearts', 'diamonds', 'clubs', 'spades'];
    const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
    const deck = [];
    let idCounter = 1;
    const ts = Date.now();
    
    for (let i = 0; i < 2; i++) {
      for (const suit of suits) {
        for (const rank of ranks) {
          deck.push({ id: `card-${ts}-${idCounter++}`, suit, rank });
        }
      }
    }
    
    deck.push({ id: `card-${ts}-${idCounter++}`, joker: 'red' });
    deck.push({ id: `card-${ts}-${idCounter++}`, joker: 'red' });
    deck.push({ id: `card-${ts}-${idCounter++}`, joker: 'black' });
    deck.push({ id: `card-${ts}-${idCounter++}`, joker: 'black' });
    
    return deck;
  };

  const shuffleDeck = (deck) => {
    const shuffled = [...deck];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  };

  const fetchMatchData = async (isBackgroundUpdate = false) => {
    if (!isBackgroundUpdate) setLoading(true);
    setError('');
    
    try {
      const matchRecord = await pb.collection('matches').getOne(matchId, {
        expand: 'lobby_id',
        $autoCancel: false
      });
      setMatch(matchRecord);

      const playersRecords = await pb.collection('matchplayers').getFullList({
        filter: `match_id = "${matchId}"`,
        expand: 'user_id',
        sort: 'seat',
        $autoCancel: false
      });
      setMatchPlayers(playersRecords);

      if (!isBackgroundUpdate) setLoading(false);
    } catch (err) {
      console.error('Fetch match error:', err);
      if (!gameWinner) {
        setError('Failed to load match data. Please try again.');
      }
      if (!isBackgroundUpdate) setLoading(false);
    }
  };

  useEffect(() => {
    fetchMatchData(false);

    const unsubscribeMatch = pb.collection('matches').subscribe(matchId, (e) => {
      if (e.action === 'delete') {
        if (!gameWinner) {
          navigate('/');
          toast.info('Match ended.');
        }
        return;
      }
      if (e.action === 'update') {
        setMatch(e.record);
        if (e.record?.deck && e.record.deck.length === 0) {
          setDeckLocked(false);
          setIsDrawing(false);
          setDrawnCard(null);
        }
        if (e.record.gameStatus === 'finished') {
          const winnerInfo = e.record.trickLastPlay?.[0]?.team;
          if (winnerInfo) {
            setGameWinner(winnerInfo);
            toast.success(`Game Over! ${winnerInfo} team wins the entire match!`);
          }
          
          const myPlayer = matchPlayersRef.current.find(p => p.user_id === currentUser?.id);
          if (myPlayer && myPlayer.seat === '1') {
            setTimeout(async () => {
              try {
                const allPlayers = await pb.collection('matchplayers').getFullList({
                  filter: `match_id = "${matchId}"`,
                  $autoCancel: false
                });
                for (const p of allPlayers) {
                  await pb.collection('matchplayers').delete(p.id, { $autoCancel: false }).catch(() => {});
                }
                await pb.collection('matches').delete(matchId, { $autoCancel: false }).catch(() => {});
                if (e.record.lobby_id) {
                  await pb.collection('lobbies').update(e.record.lobby_id, { status: 'waiting' }, { $autoCancel: false }).catch(() => {});
                }
              } catch (err) {
                console.error('Cleanup error:', err);
              }
            }, 7000);
          }
        }
      }
    }, { $autoCancel: false });

    const unsubscribeMatchPlayers = pb.collection('matchplayers').subscribe('*', (e) => {
      if (e.record?.match_id === matchId) {
        if (e.action !== 'delete') {
          fetchMatchData(true).then(() => {
            if (e.action === 'update' && e.record.user_id === currentUser?.id) {
              setTimeout(() => {
                setDeckLocked(false);
                setIsDrawing(false);
                setDrawnCard(null);
              }, 300);
            }
          });
        }
      }
    }, { 
      filter: `match_id = "${matchId}"`,
      $autoCancel: false 
    });

    return () => {
      unsubscribeMatch.then(unsub => unsub()).catch(err => console.error('Unsubscribe match error:', err));
      unsubscribeMatchPlayers.then(unsub => unsub()).catch(err => console.error('Unsubscribe matchplayers error:', err));
      if (drawTimeoutRef.current) {
        clearTimeout(drawTimeoutRef.current);
      }
    };
  }, [matchId, currentUser, gameWinner, navigate]);

  useEffect(() => {
    if (!match) return;

    const currentSeat = match.currentSeat;
    const lastTrickSeat = match.lastTrickSeat;

    if (lastTrickSeat && Number(currentSeat) === Number(lastTrickSeat)) {
      // Tunnel: when the visible trick clears, persist that clear in tunnelState too.
      const matchUpdateData = buildTunnelMatchUpdate(
        match,
        matchPlayers,
        {
          trickLastPlay: [],
          lastTrickSeat: null
        },
        {
          type: 'trick_cleared',
          seat: currentSeat,
          reason: 'turn_returned_to_leader',
          clearCurrentTrick: true
        }
      );

      pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false }).catch(err => {
        console.error('[useEffect - Trick Clearing] Error clearing trick:', err);
      });
    }
  }, [match, matchPlayers, match?.currentSeat, match?.lastTrickSeat, matchId]);

  const handleRetry = () => {
    fetchMatchData(false);
  };

  const getNextNonFinishedSeat = (currentSeatNum) => {
    let nextSeat = currentSeatNum === 4 ? 1 : currentSeatNum + 1;
    let attempts = 0;
    let shouldClearTrick = false;
    let clearedLeaderSeat = null;
    
    while (attempts < 4) {
      const nextPlayer = matchPlayers.find(p => parseInt(p.seat) === nextSeat);
      
      if (nextPlayer && nextPlayer.finishPlace) {
        const finishedPlayerSeat = parseInt(nextPlayer.seat);
        const lastTrickSeat = match?.lastTrickSeat;
        
        if (lastTrickSeat && Number(lastTrickSeat) === Number(finishedPlayerSeat)) {
          shouldClearTrick = true;
          clearedLeaderSeat = finishedPlayerSeat;
          const teammateSeat = getTeammateSeat(finishedPlayerSeat, nextPlayer.team);
          
          if (teammateSeat) {
            // Tunnel: hand the seat choice back to the pass handler so it can write one consistent match+tunnel update.
            return {
              nextSeat: teammateSeat,
              shouldClearTrick,
              clearedLeaderSeat
            };
          }
        }
      }
      
      if (nextPlayer && !nextPlayer.finishPlace) {
        return {
          nextSeat,
          shouldClearTrick,
          clearedLeaderSeat
        };
      }
      
      nextSeat = nextSeat === 4 ? 1 : nextSeat + 1;
      attempts++;
    }
    return {
      nextSeat: null,
      shouldClearTrick,
      clearedLeaderSeat
    };
  };

  const getTeammateSeat = (playerSeat, playerTeam) => {
    const teammate = matchPlayers.find(p => 
      parseInt(p.seat) !== playerSeat && 
      p.team === playerTeam && 
      !p.finishPlace
    );
    if (teammate) return parseInt(teammate.seat);
    return null;
  };

  const getNextFinishPlace = () => {
    const finishedPlayers = matchPlayers.filter(p => p.finishPlace);
    return finishedPlayers.length + 1;
  };

  const assignRemainingFinishPlaces = async (players) => {
    const finishedPlayers = players.filter(p => p.finishPlace).sort((a, b) => parseInt(a.finishPlace) - parseInt(b.finishPlace));
    const unfinishedPlayers = players.filter(p => !p.finishPlace);
    
    if (unfinishedPlayers.length === 0) return;
    
    const firstPlace = finishedPlayers.find(p => p.finishPlace === '1');
    const secondPlace = finishedPlayers.find(p => p.finishPlace === '2');
    const thirdPlace = finishedPlayers.find(p => p.finishPlace === '3');
    
    if (firstPlace && secondPlace && firstPlace.team === secondPlace.team) {
      const losingTeam = firstPlace.team === 'Blue' ? 'Red' : 'Blue';
      const losingTeamPlayers = unfinishedPlayers.filter(p => p.team === losingTeam);
      
      if (losingTeamPlayers.length === 2) {
        await pb.collection('matchplayers').update(losingTeamPlayers[0].id, { finishPlace: '3' }, { $autoCancel: false });
        await pb.collection('matchplayers').update(losingTeamPlayers[1].id, { finishPlace: '4' }, { $autoCancel: false });
      } else if (losingTeamPlayers.length === 1) {
        const nextPlace = thirdPlace ? '4' : '3';
        await pb.collection('matchplayers').update(losingTeamPlayers[0].id, { finishPlace: nextPlace }, { $autoCancel: false });
      }
    } else if (unfinishedPlayers.length === 1) {
      await pb.collection('matchplayers').update(unfinishedPlayers[0].id, { finishPlace: '4' }, { $autoCancel: false });
    }
  };

  const checkTeamWin = async (updatedPlayers) => {
    const playersToCheck = updatedPlayers || matchPlayers;
    const finishedPlayers = playersToCheck.filter(p => p.finishPlace).sort((a, b) => parseInt(a.finishPlace) - parseInt(b.finishPlace));
    
    if (finishedPlayers.length < 2) return false;
    
    const firstPlace = finishedPlayers.find(p => p.finishPlace === '1');
    const secondPlace = finishedPlayers.find(p => p.finishPlace === '2');
    
    if (!firstPlace || !secondPlace) return false;
    
    if (firstPlace.team === secondPlace.team) {
      await assignRemainingFinishPlaces(playersToCheck);
      await endRound(firstPlace.team);
      return true;
    }
    return false;
  };

  const checkRoundEnd = async (updatedPlayers) => {
    const playersToCheck = updatedPlayers || matchPlayers;
    const nonFinishedPlayers = playersToCheck.filter(p => !p.finishPlace);
    
    if (nonFinishedPlayers.length === 1) {
      const lastPlayer = nonFinishedPlayers[0];
      await pb.collection('matchplayers').update(lastPlayer.id, { finishPlace: '4' }, { $autoCancel: false });
      
      const finalPlayers = await pb.collection('matchplayers').getFullList({
        filter: `match_id = "${matchId}"`,
        expand: 'user_id',
        sort: 'seat',
        $autoCancel: false
      });
      
      const finishedPlayers = finalPlayers.filter(p => p.finishPlace).sort((a, b) => parseInt(a.finishPlace) - parseInt(b.finishPlace));
      const firstPlace = finishedPlayers.find(p => p.finishPlace === '1');
      
      if (firstPlace) {
        await endRound(firstPlace.team);
      }
      return true;
    }
    return false;
  };

  const endRound = async (winningTeam) => {
    try {
      const currentMatch = await pb.collection('matches').getOne(matchId, { $autoCancel: false });
      const allPlayers = await pb.collection('matchplayers').getFullList({
        filter: `match_id = "${matchId}"`,
        expand: 'user_id',
        sort: 'seat',
        $autoCancel: false
      });
      
      const finishedPlayers = allPlayers.filter(p => p.finishPlace).sort((a, b) => parseInt(a.finishPlace) - parseInt(b.finishPlace));
      const firstPlace = finishedPlayers.find(p => p.finishPlace === '1');
      const secondPlace = finishedPlayers.find(p => p.finishPlace === '2');
      const thirdPlace = finishedPlayers.find(p => p.finishPlace === '3');
      const fourthPlace = finishedPlayers.find(p => p.finishPlace === '4');
      
      let rankIncrease = 1;
      if (firstPlace && secondPlace && firstPlace.team === secondPlace.team) {
        rankIncrease = 3;
      } else if (firstPlace && thirdPlace && firstPlace.team === thirdPlace.team) {
        rankIncrease = 2;
      } else if (firstPlace && fourthPlace && firstPlace.team === fourthPlace.team) {
        rankIncrease = 1;
      }
      
      const currentBlueRank = currentMatch.levelRankBlue || '2';
      const currentRedRank = currentMatch.levelRankRed || '2';
      const rankOrder = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A1', 'A2', 'A3'];
      
      const maxNormalIndex = rankOrder.indexOf('A1');
      
      let newBlueRank = currentBlueRank;
      let newRedRank = currentRedRank;
      let newCaller = winningTeam;
      let isGameWon = false;

      const callerTeam = currentMatch.currentRoundLevelRank || 'Blue';
      const callerRank = callerTeam === 'Blue' ? currentBlueRank : currentRedRank;

      if (callerRank.startsWith('A')) {
        if (winningTeam === callerTeam) {
          if (rankIncrease >= 2) {
            isGameWon = true;
          } else {
            newCaller = callerTeam;
            if (callerTeam === 'Blue') {
              newBlueRank = currentBlueRank === 'A1' ? 'A2' : 'A3';
              newRedRank = currentRedRank;
            } else {
              newRedRank = currentRedRank === 'A1' ? 'A2' : 'A3';
              newBlueRank = currentBlueRank;
            }
          }
        } else {
          if (callerTeam === 'Blue') {
            newBlueRank = currentBlueRank === 'A1' ? 'A2' : currentBlueRank === 'A2' ? 'A3' : '2';
            const currentIndex = rankOrder.indexOf(currentRedRank);
            newRedRank = rankOrder[Math.min(currentIndex + rankIncrease, maxNormalIndex)];
          } else {
            newRedRank = currentRedRank === 'A1' ? 'A2' : currentRedRank === 'A2' ? 'A3' : '2';
            const currentIndex = rankOrder.indexOf(currentBlueRank);
            newBlueRank = rankOrder[Math.min(currentIndex + rankIncrease, maxNormalIndex)];
          }
          newCaller = winningTeam;
        }
      } else {
        if (winningTeam === 'Blue') {
          const currentIndex = rankOrder.indexOf(currentBlueRank);
          newBlueRank = rankOrder[Math.min(currentIndex + rankIncrease, maxNormalIndex)];
        } else {
          const currentIndex = rankOrder.indexOf(currentRedRank);
          newRedRank = rankOrder[Math.min(currentIndex + rankIncrease, maxNormalIndex)];
        }
        newCaller = winningTeam;
      }

      if (isGameWon) {
        // Tunnel: mark the match as finished without leaving a phantom current trick in the ledger.
        const matchUpdateData = buildTunnelMatchUpdate(
          currentMatch,
          allPlayers,
          {
            gameStatus: 'finished',
            trickLastPlay: [{ team: winningTeam }]
          },
          {
            type: 'match_finished',
            winningTeam,
            clearCurrentTrick: true
          }
        );

        await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
        return;
      }

      const newRoundNumber = (currentMatch.roundNumber || 1) + 1;
      let newDeck = createDeck();
      newDeck = shuffleDeck(newDeck);
      
      const firstFinisherSeat = parseInt(firstPlace.seat);
      const dealingStartSeat = (firstFinisherSeat % 4) + 1;
      
      for (const player of allPlayers) {
        await pb.collection('matchplayers').update(player.id, {
          hand: [],
          groups: []
        }, { $autoCancel: false });
      }
      
      // Tunnel: a new round gets a fresh roundKey plus a bootstrap event for the bot ledger.
      const matchUpdateData = buildTunnelMatchUpdate(currentMatch, allPlayers, {
        gameStatus: 'dealing',
        currentRoundLevelRank: newCaller,
        levelRankBlue: newBlueRank,
        levelRankRed: newRedRank,
        roundNumber: newRoundNumber,
        deck: newDeck,
        trickLastPlay: [],
        lastTrickSeat: null,
        currentSeat: dealingStartSeat.toString(),
        startSeat: null
      }, {
        type: 'round_reset',
        winningTeam,
        currentSeat: dealingStartSeat,
        clearCurrentTrick: true
      });
      
      await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
      toast.success(`${winningTeam} team wins round ${currentMatch.roundNumber}! Next round starting.`);
      
    } catch (err) {
      console.error('[endRound] ERROR during round end:', err);
      toast.error('Failed to end round.');
    }
  };

  const handleDrawCard = async () => {
    if (!match || !currentUser || isDrawing || deckLocked) return;

    const currentPlayerRecord = matchPlayers.find(p => p.seat === match.currentSeat);
    
    if (!currentPlayerRecord || currentPlayerRecord.user_id !== currentUser.id) {
      toast.error('Not your turn');
      return;
    }

    if (!match.deck || match.deck.length === 0) return;

    const drawnCardRaw = match.deck[0];
    const cleanCard = { ...drawnCardRaw };
    delete cleanCard.isFaceUp;
    
    if (!cleanCard.id) {
      cleanCard.id = `draw-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    }

    setDrawnCard(cleanCard);
    setIsDrawing(true);
    setDeckLocked(true);

    try {
      const remainingDeck = match.deck.slice(1);
      const currentHand = currentPlayerRecord.hand || [];
      const isFaceUpCard = drawnCardRaw.isFaceUp === true;
      
      const updatedHand = [...currentHand, cleanCard];
      const currentSeatNum = parseInt(match.currentSeat);
      const nextSeat = currentSeatNum === 4 ? 1 : currentSeatNum + 1;

      let matchUpdateData = { deck: remainingDeck };

      if (isFaceUpCard) matchUpdateData.startSeat = currentSeatNum;

      if (remainingDeck.length === 0) {
        const finalStartSeat = matchUpdateData.startSeat || match.startSeat;
        const currentRound = match.roundNumber || 1;

        if (currentRound === 1) {
          matchUpdateData.gameStatus = 'playing';
        } else {
          matchUpdateData.gameStatus = 'tribute';
        }
        matchUpdateData.currentSeat = finalStartSeat ? finalStartSeat.toString() : '1';
      } else {
        matchUpdateData.gameStatus = 'dealing';
        matchUpdateData.currentSeat = nextSeat.toString();
      }

      // Tunnel: sync the phase transition into tunnelState once the dealing step updates the match.
      const tunnelEvent = remainingDeck.length === 0
        ? {
            type: 'phase_sync',
            phase: matchUpdateData.gameStatus,
            reason: 'deal_finished',
            currentSeat: matchUpdateData.currentSeat
          }
        : null;
      matchUpdateData = buildTunnelMatchUpdate(match, matchPlayers, matchUpdateData, tunnelEvent);

      await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
      await pb.collection('matchplayers').update(currentPlayerRecord.id, { hand: updatedHand }, { $autoCancel: false });

      if (isFaceUpCard) toast.success('You are the start player');

      drawTimeoutRef.current = setTimeout(() => {
        const updatedPlayer = matchPlayers.find(p => p.id === currentPlayerRecord.id);
        if (!updatedPlayer || updatedPlayer.hand.length !== updatedHand.length) {
          fetchMatchData(true);
        }
      }, 2000);

    } catch (err) {
      console.error('Draw card error:', err);
      toast.error('Failed to draw card. Please try again.');
      setIsDrawing(false);
      setDeckLocked(false);
      setDrawnCard(null);
    }
  };

  const handlePlayCards = async (cardsToPlay) => {
    if (!match || !currentUser || isPlaying) return;

    const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
    if (!currentPlayerRecord || match.currentSeat !== currentPlayerRecord.seat) return;

    const currentRoundLevelRank = getCurrentLevelRank();
    const currentSeatNum = parseInt(match.currentSeat);
    const lastTrickSeat = match.lastTrickSeat;
    
    let shouldClearTrick = false;
    if (lastTrickSeat && Number(currentSeatNum) === Number(lastTrickSeat)) {
      shouldClearTrick = true;
    }
    
    const validation = validatePlay(cardsToPlay, shouldClearTrick ? [] : (match.trickLastPlay || []), currentRoundLevelRank);
    
    if (!validation.isValid) {
      toast.error(validation.error);
      return;
    }

    setIsPlaying(true);

    try {
      const currentHand = currentPlayerRecord?.hand || [];
      const updatedHand = [...currentHand];
      const playedCardIds = new Set(cardsToPlay.map(c => c.id));

      for (const playCard of cardsToPlay) {
        const index = updatedHand.findIndex(handCard => {
          if (playCard.id && handCard.id) {
            return handCard.id === playCard.id;
          }
          return handCard.joker === playCard.joker &&
                 handCard.suit === playCard.suit &&
                 handCard.rank === playCard.rank;
        });
        if (index !== -1) {
          updatedHand.splice(index, 1);
        }
      }

      // Clean up groups: remove played cards from groups, and remove groups with < 2 cards
      const currentGroups = currentPlayerRecord.groups || [];
      const updatedGroups = currentGroups.map(g => ({
        ...g,
        cardIds: g.cardIds.filter(id => !playedCardIds.has(id))
      })).filter(g => g.cardIds.length > 1);

      await pb.collection('matchplayers').update(currentPlayerRecord.id, {
        hand: updatedHand,
        groups: updatedGroups
      }, { $autoCancel: false });

      if (updatedHand.length === 0) {
        const nextFinishPlace = getNextFinishPlace();
        await pb.collection('matchplayers').update(currentPlayerRecord.id, {
          finishPlace: nextFinishPlace.toString()
        }, { $autoCancel: false });
        
        toast.success(`You finished in place ${nextFinishPlace}!`);
        
        const updatedPlayers = await pb.collection('matchplayers').getFullList({
          filter: `match_id = "${matchId}"`,
          expand: 'user_id',
          sort: 'seat',
          $autoCancel: false
        });
        
        setMatchPlayers(updatedPlayers);
        
        const teamWon = await checkTeamWin(updatedPlayers);
        if (teamWon) {
          setIsPlaying(false);
          return;
        }
        
        const roundEnded = await checkRoundEnd(updatedPlayers);
        if (roundEnded) {
          setIsPlaying(false);
          return;
        }
        
        let nextSeat = currentSeatNum === 4 ? 1 : currentSeatNum + 1;
        let rotationAttempts = 0;
        while (rotationAttempts < 4) {
          const nextPlayer = updatedPlayers.find(p => parseInt(p.seat) === nextSeat);
          if (nextPlayer && !nextPlayer.finishPlace) break;
          nextSeat = nextSeat === 4 ? 1 : nextSeat + 1;
          rotationAttempts++;
        }
        
        if (rotationAttempts >= 4) {
          setIsPlaying(false);
          return;
        }
        
        // Tunnel: append the full play event so a single bot can still replay every trick from PocketBase.
        const matchUpdateData = buildTunnelMatchUpdate(
          match,
          updatedPlayers,
          {
            trickLastPlay: cardsToPlay,
            lastTrickSeat: currentSeatNum,
            currentSeat: nextSeat.toString()
          },
          shouldClearTrick
            ? [
                {
                  type: 'trick_cleared',
                  seat: currentSeatNum,
                  reason: 'leader_reopened',
                  clearCurrentTrick: true,
                  currentSeat: currentSeatNum
                },
                {
                  type: 'play',
                  seat: currentSeatNum,
                  cards: cardsToPlay,
                  currentSeat: nextSeat,
                  finishOrder: updatedPlayers.filter(player => player.finishPlace).map(player => player.seat)
                }
              ]
            : {
                type: 'play',
                seat: currentSeatNum,
                cards: cardsToPlay,
                currentSeat: nextSeat,
                finishOrder: updatedPlayers.filter(player => player.finishPlace).map(player => player.seat)
              }
        );

        await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
      } else {
        const updatedPlayers = await pb.collection('matchplayers').getFullList({
          filter: `match_id = "${matchId}"`,
          expand: 'user_id',
          sort: 'seat',
          $autoCancel: false
        });
        
        let nextSeat = currentSeatNum === 4 ? 1 : currentSeatNum + 1;
        let rotationAttempts = 0;
        
        while (rotationAttempts < 4) {
          const nextPlayer = updatedPlayers.find(p => parseInt(p.seat) === nextSeat);
          if (nextPlayer && !nextPlayer.finishPlace) break;
          nextSeat = nextSeat === 4 ? 1 : nextSeat + 1;
          rotationAttempts++;
        }
        
        if (rotationAttempts >= 4) {
          setIsPlaying(false);
          return;
        }
        
        // Tunnel: normal plays hit the same event ledger so the Python bridge sees the complete trick flow.
        const matchUpdateData = buildTunnelMatchUpdate(
          match,
          updatedPlayers,
          {
            trickLastPlay: cardsToPlay,
            lastTrickSeat: currentSeatNum,
            currentSeat: nextSeat.toString()
          },
          shouldClearTrick
            ? [
                {
                  type: 'trick_cleared',
                  seat: currentSeatNum,
                  reason: 'leader_reopened',
                  clearCurrentTrick: true,
                  currentSeat: currentSeatNum
                },
                {
                  type: 'play',
                  seat: currentSeatNum,
                  cards: cardsToPlay,
                  currentSeat: nextSeat,
                  finishOrder: updatedPlayers.filter(player => player.finishPlace).map(player => player.seat)
                }
              ]
            : {
                type: 'play',
                seat: currentSeatNum,
                cards: cardsToPlay,
                currentSeat: nextSeat,
                finishOrder: updatedPlayers.filter(player => player.finishPlace).map(player => player.seat)
              }
        );

        await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
      }

      const combo = detectCombination(cardsToPlay, currentRoundLevelRank);
      if (isBomb(combo)) {
        toast.success(`BOMB played: ${combo.type.replace(/_/g, ' ')}`);
      } else {
        toast.success(`Played ${combo.type.toLowerCase().replace(/_/g, ' ')}`);
      }
      setSelectedCards([]);

    } catch (err) {
      console.error('[handlePlayCards] ERROR:', err);
      toast.error('Failed to play cards. Please try again.');
    } finally {
      setIsPlaying(false);
    }
  };

  const handlePass = async () => {
    if (!match || !currentUser) return;
    const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
    if (!currentPlayerRecord || match.currentSeat !== currentPlayerRecord.seat) return;
    if (!match.trickLastPlay || match.trickLastPlay.length === 0) return;

    try {
      const currentSeatNum = parseInt(match.currentSeat);
      const { nextSeat, shouldClearTrick, clearedLeaderSeat } = getNextNonFinishedSeat(currentSeatNum);
      
      // Tunnel: passes matter for replaying the live trick order, so persist them explicitly.
      const matchUpdateData = buildTunnelMatchUpdate(
        match,
        matchPlayers,
        {
          currentSeat: nextSeat ? nextSeat.toString() : match.currentSeat,
          ...(shouldClearTrick ? { trickLastPlay: [], lastTrickSeat: null } : {})
        },
        shouldClearTrick
          ? [
              {
                type: 'pass',
                seat: currentSeatNum,
                currentSeat: nextSeat || match.currentSeat
              },
              {
                type: 'trick_cleared',
                seat: clearedLeaderSeat || currentSeatNum,
                reason: 'leader_finished',
                clearCurrentTrick: true,
                currentSeat: nextSeat || match.currentSeat
              }
            ]
          : {
              type: 'pass',
              seat: currentSeatNum,
              currentSeat: nextSeat || match.currentSeat
            }
      );

      await pb.collection('matches').update(matchId, matchUpdateData, { $autoCancel: false });
      
      toast.success('Passed');
    } catch (err) {
      console.error('Pass error:', err);
      toast.error('Failed to pass. Please try again.');
    }
  };

  const handleGroupsChange = async (newGroups) => {
    if (!match || !currentUser) return;
    const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
    if (!currentPlayerRecord) return;

    const updatedPlayers = matchPlayers.map(p => 
      p.id === currentPlayerRecord.id ? { ...p, groups: newGroups } : p
    );
    setMatchPlayers(updatedPlayers);

    try {
      await pb.collection('matchplayers').update(currentPlayerRecord.id, {
        groups: newGroups
      }, { $autoCancel: false });
    } catch (err) {
      console.error('Failed to update groups:', err);
      toast.error('Failed to save groups.');
    }
  };

  const getSeatPositionClass = (seatNumber) => {
    const seatNum = parseInt(seatNumber);
    switch (seatNum) {
      case 1: return 'seat-box-top-right';
      case 2: return 'seat-box-top-left';
      case 3: return 'seat-box-bottom-left';
      case 4: return 'seat-box-bottom-right';
      default: return 'seat-box-top-right';
    }
  };

  const getPlayerAvatar = (player) => {
    const user = player.expand?.user_id;
    if (user && user.avatar) {
      return pb.files.getUrl(user, user.avatar, { thumb: '100x100' });
    }
    return null;
  };

  const getPlayerInitial = (player) => {
    const user = player.expand?.user_id;
    return user && user.username ? user.username.charAt(0).toUpperCase() : '?';
  };

  const isCurrentPlayer = (player) => match?.currentSeat === player.seat;
  const isStartPlayer = (player) => match?.startSeat && match.startSeat.toString() === player.seat;

  const canDrawCard = () => {
    if (!match || !currentUser || isDrawing || deckLocked) return false;
    const currentPlayerRecord = matchPlayers.find(p => p.seat === match.currentSeat);
    return currentPlayerRecord && currentPlayerRecord.user_id === currentUser.id;
  };

  const getCurrentPlayerHand = () => {
    if (!currentUser || !matchPlayers || matchPlayers.length === 0) return [];
    const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
    return currentPlayerRecord?.hand || [];
  };

  const getCurrentLevelRank = () => {
    if (!match) return '2';
    return match.currentRoundLevelRank === 'Blue' ? (match.levelRankBlue || '2') : (match.levelRankRed || '2');
  };

  const isMyTurn = () => {
    if (!match || !currentUser || !matchPlayers || matchPlayers.length === 0) return false;
    const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
    return currentPlayerRecord && match.currentSeat === currentPlayerRecord.seat;
  };

  const canPlay = () => {
    if (!isMyTurn() || isPlaying || selectedCards.length === 0) return false;
    
    const currentRoundLevelRank = getCurrentLevelRank();
    const combo = detectCombination(selectedCards, currentRoundLevelRank);
    if (!combo || !combo.isValid) return false;
    
    const currentSeatNum = parseInt(match.currentSeat);
    const lastTrickSeat = match.lastTrickSeat;
    const shouldClearTrick = lastTrickSeat && Number(currentSeatNum) === Number(lastTrickSeat);
    const trickLastPlay = shouldClearTrick ? [] : (match?.trickLastPlay || []);
    
    const validation = validatePlay(selectedCards, trickLastPlay, currentRoundLevelRank);
    return validation.isValid;
  };

  const canPass = () => {
    return isMyTurn() && match?.trickLastPlay && match.trickLastPlay.length > 0;
  };

  if (gameWinner) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex items-center justify-center px-4">
        <Helmet>
          <title>Game Over - Guandan</title>
        </Helmet>
        <Card className="max-w-md w-full shadow-2xl border-primary/20">
          <CardHeader className="text-center pb-2">
            <CardTitle className="text-4xl font-extrabold tracking-tight">Game Over</CardTitle>
            <CardDescription className="text-lg mt-2">The match has concluded</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center space-y-8 pt-6">
            <div className="flex flex-col items-center justify-center space-y-2">
              <span className="text-6xl" role="img" aria-label="Trophy">🏆</span>
              <h2 className={`text-4xl font-black uppercase tracking-widest ${gameWinner === 'Blue' ? 'text-blue-500' : 'text-red-500'}`}>
                {gameWinner} Wins
              </h2>
            </div>
            
            <div className="w-full bg-muted/50 rounded-xl p-4 grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Blue Team</p>
                <p className="text-3xl font-bold text-blue-500">{match?.levelRankBlue}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Red Team</p>
                <p className="text-3xl font-bold text-red-500">{match?.levelRankRed}</p>
              </div>
            </div>

            <Button 
              size="lg" 
              className="w-full text-lg h-14" 
              onClick={() => navigate('/')}
            >
              Return to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading match...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background px-4">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>Error</CardTitle>
            <CardDescription>Failed to load match</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">{error}</p>
            <Button onClick={handleRetry} className="w-full gap-2">
              <RefreshCw className="w-4 h-4" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const playButtonDisabled = !canPlay();
  const currentPlayerRecord = matchPlayers.find(p => p.user_id === currentUser.id);
  const handMode = match?.gameStatus === 'tribute' ? 'tribute' : 'play';
  const lastTrickPlayer = matchPlayers?.find(p => p.seat === match?.lastTrickSeat?.toString());
  const lastTrickUsername = lastTrickPlayer?.expand?.user_id?.username || `Seat ${match?.lastTrickSeat}`;
  // Tunnel: surface the durable tunnel pass counter in the trick UI when it exists.
  const tunnelPassCount = match?.tunnelState?.currentTrick?.passCount ?? match.passCount;

  return (
    <>
      <Helmet>
        <title>Match - Guandan Card Game</title>
        <meta name="description" content="Guandan card game match in progress" />
      </Helmet>

      <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex flex-col relative">
        {isDrawing && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/60 backdrop-blur-sm">
            <div className="bg-card p-8 rounded-2xl shadow-2xl border border-border flex flex-col items-center gap-6 animate-in fade-in zoom-in duration-200">
              <h3 className="text-2xl font-bold text-foreground">You drew</h3>
              {drawnCard ? (
                <div className="transform scale-150 my-6 pointer-events-none">
                  <CardDisplay card={drawnCard} />
                </div>
              ) : (
                <Loader2 className="w-12 h-12 text-primary animate-spin my-6" />
              )}
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" />
                <p className="animate-pulse">Adding to hand...</p>
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="game-board-container">
              <div className="team-info-box">
                <div className="team-info-scores">
                  <div 
                    className={`team-score-badge team-score-badge-blue transition-all duration-300 ${
                      match?.currentRoundLevelRank === 'Blue' 
                        ? 'ring-2 ring-offset-2 ring-blue-400 ring-offset-background z-10 shadow-lg font-bold' 
                        : 'opacity-95 scale-95 font-bold'
                    }`}
                  >
                    Blue: {match?.levelRankBlue || '2'}
                  </div>
                  <div 
                    className={`team-score-badge team-score-badge-red transition-all duration-300 ${
                      match?.currentRoundLevelRank === 'Red' 
                        ? 'ring-2 ring-offset-2 ring-red-400 ring-offset-background z-10 shadow-lg font-bold' 
                        : 'opacity-95 scale-95 font-bold'
                    }`}
                  >
                    Red: {match?.levelRankRed || '2'}
                  </div>
                </div>
              </div>

              {match?.deck?.length > 0 ? (
                <DeckCard 
                  cardCount={match.deck.length}
                  onClick={handleDrawCard}
                  disabled={!canDrawCard()}
                  isDrawing={isDrawing}
                  topCard={match.deck[0]}
                />
              ) : null}

              {match?.gameStatus === 'tribute' && (
                <TributePhase 
                  currentMatch={match}
                  currentPlayer={currentPlayerRecord}
                  allMatchPlayers={matchPlayers}
                  currentUser={currentUser}
                  onCardSelect={setSelectedCards}
                  selectedCards={selectedCards}
                  tributeRef={tributeRef}
                />
              )}

              {match?.gameStatus === 'playing' && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl px-4">
                  <TrickDisplay 
                    trickLastPlay={match.trickLastPlay}
                    lastTrickUsername={lastTrickUsername} 
                    passCount={tunnelPassCount}
                    currentRoundLevelRank={getCurrentLevelRank()}
                  />
                </div>
              )}

              {matchPlayers && matchPlayers.length > 0 && matchPlayers.map((player) => {
                const avatarUrl = getPlayerAvatar(player);
                const initial = getPlayerInitial(player);
                const username = player.expand?.user_id?.username || 'Unknown';
                const teamClass = player.team === 'Blue' ? 'seat-box-blue' : 'seat-box-red';
                const positionClass = getSeatPositionClass(player.seat);
                const activeClass = isCurrentPlayer(player) ? 'seat-box-active' : '';
                const showStartBadge = match?.gameStatus === 'dealing' && !!isStartPlayer(player);
                const finishPlace = player.finishPlace;

                // 1. Get the card count (Adjust 'cardCount' or 'hand' to match your actual database schema)
                const cardCount = player.cardCount ?? player.hand?.length ?? 0;

                return (
                  <div
                    key={player.id}
                    className={`seat-box ${teamClass} ${positionClass} ${activeClass}`}
                  >
                    {showStartBadge && (
                      <div className="absolute -top-2 -right-2 bg-primary text-primary-foreground text-[10px] font-bold px-2 py-0.5 rounded-full z-5 shadow-md">
                        START
                      </div>
                    )}
                    {typeof finishPlace === 'string' && finishPlace.trim() !== '' && (
                      <div className="absolute -top-2 -left-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full z-5 shadow-md">
                        #{finishPlace}
                      </div>
                    )}

                    {/* 2. Add the conditional card count badge here */}
                    {cardCount > 0 && cardCount <= 10 && (
                      <div className="absolute -bottom-2 -right-2 bg-red-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full z-10 shadow-md animate-bounce">
                        {cardCount} Cards
                      </div>
                    )}

                    <div className="seat-player-avatar">
                      {avatarUrl ? (
                        <img src={avatarUrl} alt={username} />
                      ) : (
                        <span>{initial}</span>
                      )}
                    </div>
                    <div className="seat-player-name">{username}</div>
                    <div className="seat-number">Seat {player.seat}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {match?.gameStatus === 'playing' && isMyTurn() && (
          <div className="bg-card border-t border-border py-3 px-4">
            <div className="max-w-7xl mx-auto flex items-center justify-center gap-3">
              <Button
                onClick={() => handlePlayCards(selectedCards)}
                disabled={playButtonDisabled}
                size="lg"
                className="gap-2"
              >
                <Play className="w-5 h-5" />
                Play Cards
              </Button>
              <Button
                onClick={handlePass}
                disabled={!canPass()}
                variant="outline"
                size="lg"
                className="gap-2"
              >
                <SkipForward className="w-5 h-5" />
                Pass
              </Button>
            </div>
          </div>
        )}

        <CurrentPlayerHand 
          cards={getCurrentPlayerHand()} 
          levelRank={getCurrentLevelRank()}
          onPlayCards={setSelectedCards}
          mode={handMode}
          tributeRef={tributeRef}
          groups={currentPlayerRecord?.groups || []}
          onGroupsChange={handleGroupsChange}
        />
      </div>
    </>
  );
};

export default MatchPage;
