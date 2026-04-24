export const TUNNEL_VERSION = 1;

export const sanitizeTunnelCard = (card = {}) => {
  const clean = {};

  if (card?.id !== undefined && card?.id !== null) {
    clean.id = card.id;
  }

  if (card?.joker) {
    clean.joker = card.joker;
    return clean;
  }

  if (card?.suit) {
    clean.suit = card.suit;
  }

  if (card?.rank) {
    clean.rank = card.rank;
  }

  return clean;
};

const sanitizeTunnelCards = (cards = []) => {
  if (!Array.isArray(cards)) return [];
  return cards.map(sanitizeTunnelCard);
};

const sameTunnelCards = (cardsA = [], cardsB = []) => {
  if (cardsA.length !== cardsB.length) return false;

  return cardsA.every((card, index) => {
    const other = cardsB[index];
    if (!other) return false;

    if (card.id && other.id) {
      return card.id === other.id;
    }

    return card.joker === other.joker &&
      card.suit === other.suit &&
      card.rank === other.rank;
  });
};

export const buildTunnelRoundKey = (matchLike = {}) => {
  return [
    matchLike?.roundNumber || 1,
    matchLike?.currentRoundLevelRank || 'Blue',
    matchLike?.levelRankBlue || '2',
    matchLike?.levelRankRed || '2'
  ].join(':');
};

export const buildTunnelFinishOrder = (players = []) => {
  return [...players]
    .filter(player => player?.finishPlace)
    .sort((a, b) => Number(a.finishPlace) - Number(b.finishPlace))
    .map(player => String(player.seat));
};

export const ensureTunnelState = (matchLike = {}, players = []) => {
  const existing = matchLike?.tunnelState || matchLike?.botTunnel || {};
  const roundKey = buildTunnelRoundKey(matchLike);
  const sameRound = existing?.roundKey === roundKey;
  const matchTrick = matchLike?.gameStatus === 'playing'
    ? sanitizeTunnelCards(matchLike?.trickLastPlay || [])
    : [];
  const existingTrick = sanitizeTunnelCards(existing?.currentTrick?.cards || []);

  return {
    version: TUNNEL_VERSION,
    roundKey,
    phase: matchLike?.gameStatus || 'dealing',
    finishOrder: buildTunnelFinishOrder(players),
    events: sameRound && Array.isArray(existing?.events) ? [...existing.events] : [],
    currentTrick: matchTrick.length > 0
      ? {
          leaderSeat: matchLike?.lastTrickSeat != null ? String(matchLike.lastTrickSeat) : null,
          cards: matchTrick,
          passCount: sameRound && sameTunnelCards(existingTrick, matchTrick)
            ? Number(existing?.currentTrick?.passCount || 0)
            : 0
        }
      : {
          leaderSeat: null,
          cards: [],
          passCount: 0
        }
  };
};

export const appendTunnelEvent = (matchLike = {}, players = [], event = {}) => {
  const tunnelState = ensureTunnelState(matchLike, players);
  const events = [...(tunnelState.events || [])];
  const nextSeq = events.length > 0 ? Number(events[events.length - 1]?.seq || 0) + 1 : 1;

  const tunnelEvent = {
    seq: nextSeq,
    type: event?.type || 'sync',
    phase: event?.phase || matchLike?.gameStatus || tunnelState.phase,
    currentSeat: event?.currentSeat != null
      ? String(event.currentSeat)
      : (matchLike?.currentSeat != null ? String(matchLike.currentSeat) : null),
    timestamp: Date.now(),
    ...event
  };

  if (tunnelEvent.seat != null) {
    tunnelEvent.seat = String(tunnelEvent.seat);
  }

  if (tunnelEvent.leaderSeat != null) {
    tunnelEvent.leaderSeat = String(tunnelEvent.leaderSeat);
  }

  if (Array.isArray(tunnelEvent.cards)) {
    tunnelEvent.cards = sanitizeTunnelCards(tunnelEvent.cards);
  }

  if (Array.isArray(tunnelEvent.finishOrder)) {
    tunnelEvent.finishOrder = tunnelEvent.finishOrder.map(value => String(value));
  }

  events.push(tunnelEvent);
  tunnelState.events = events;
  tunnelState.phase = tunnelEvent.phase;
  tunnelState.finishOrder = tunnelEvent.finishOrder || buildTunnelFinishOrder(players);

  if (tunnelEvent.type === 'play') {
    tunnelState.currentTrick = {
      leaderSeat: tunnelEvent.seat || null,
      cards: sanitizeTunnelCards(tunnelEvent.cards || []),
      passCount: 0
    };
  } else if (tunnelEvent.type === 'pass' && (tunnelState.currentTrick?.cards || []).length > 0) {
    tunnelState.currentTrick = {
      ...(tunnelState.currentTrick || {}),
      passCount: Number(tunnelState.currentTrick?.passCount || 0) + 1
    };
  } else if (
    tunnelEvent.type === 'trick_cleared' ||
    tunnelEvent.type === 'round_reset' ||
    tunnelEvent.type === 'phase_reset' ||
    tunnelEvent.type === 'match_finished' ||
    tunnelEvent.clearCurrentTrick
  ) {
    tunnelState.currentTrick = {
      leaderSeat: null,
      cards: [],
      passCount: 0
    };
  }

  return tunnelState;
};

export const buildTunnelMatchUpdate = (matchLike = {}, players = [], updateData = {}, event = null) => {
  const mergedMatch = {
    ...(matchLike || {}),
    ...(updateData || {})
  };
  const events = Array.isArray(event)
    ? event.filter(Boolean)
    : (event ? [event] : []);
  let tunnelState = ensureTunnelState(mergedMatch, players);

  if (events.length > 0) {
    for (const tunnelEvent of events) {
      tunnelState = appendTunnelEvent(
        {
          ...mergedMatch,
          tunnelState
        },
        players,
        tunnelEvent
      );
    }
  }

  return {
    ...(updateData || {}),
    tunnelState
  };
};
