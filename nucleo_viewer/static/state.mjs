/** Small, framework-free guards for async UI work that may finish out of order. */
export function createRaceGuards() {
  let selection = 0;
  let image = 0;

  return {
    beginSelection() {
      selection += 1;
      image += 1; // invalidate any image belonging to the previous selection
      return selection;
    },
    isCurrentSelection(ticket) {
      return ticket === selection;
    },
    beginImage(selectionTicket) {
      image += 1;
      return { selection: selectionTicket, image };
    },
    isCurrentImage(ticket) {
      return ticket.selection === selection && ticket.image === image;
    },
    currentSelection() {
      return selection;
    },
  };
}
