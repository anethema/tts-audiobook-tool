import json

import pytest

from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.speaker_assign.combined_analysis import InvalidChunkReplyError, parse_delivery_reply


def test_delivery_only_reply_cannot_change_speakers_and_records_assessed_quotes():
    reply = json.dumps({
        "delivery": {"8": {"direction": "with relief"}},
        "speakers": {"8": "WRONG"}, "characters": [{"id": "WRONG"}], "uncertain": [8],
    })
    parsed = parse_delivery_reply(reply, {3, 8})
    assert parsed.delivery == {3: DeliveryCue(), 8: DeliveryCue("with relief")}
    assert not parsed.speakers and not parsed.uncertain


@pytest.mark.parametrize("reply", [
    '{"speakers":{}}',
    '{"delivery":{"1":{},"1":{"direction":"softly"}}}',
    '{"delivery":{"2":{"direction":"softly"}}}',
    '{"delivery":{"1":"softly"}}',
])
def test_bad_delivery_only_reply_keeps_chunk_available_for_retry(reply):
    with pytest.raises(InvalidChunkReplyError):
        parse_delivery_reply(reply, {1})
