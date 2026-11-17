You check whether a claim is supported by source passages.

CLAIM
$claim

PASSAGES
$passages

Is the claim supported by the passages?
- "supported" means a passage states the same thing or something that clearly implies it. General physics that a passage explains supports the same general statement. Applying a passage's general statement to a specific event is fine if it adds no new facts.
- "not_supported" if the claim adds a cause, a number, a place, a date or an outcome that no passage gives.
- Storm level matters. If the claim is for a storm at a given NOAA G level and the passage gives the effect only for a HIGHER G level, the claim is "not_supported". For example, an effect listed under G5 does not support the same effect for a G4 storm.
- In "quote", copy the one sentence from the passages that supports the claim, word for word. If nothing supports it, leave "quote" empty.

Answer with JSON only: {"verdict": "supported" or "not_supported", "quote": "exact sentence from a passage, or empty", "reason": "one short sentence"}
