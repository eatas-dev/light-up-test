import { useState } from "react";
import { IconButton } from "@fluentui/react";

interface Props {
    answer: string;
}

const SpeechSynthesis = (window as any).speechSynthesis || (window as any).webkitSpeechSynthesis;

let synth: SpeechSynthesis | null = null;

try {
    synth = SpeechSynthesis;
} catch (err) {
    console.error("音声合成はサポートされていません");
}

const getUtterance = function (text: string) {
    if (synth) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "ja-JP";
        utterance.volume = 1;
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.voice = synth.getVoices().filter((voice: SpeechSynthesisVoice) => voice.lang === "ja-JP")[0];
        return utterance;
    }
};

export const SpeechOutput = ({ answer }: Props) => {
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    if (!synth) {
        return <></>;
    }

    const startOrStopSpeech = (answer: string) => {
        if (synth != null) {
            if (isPlaying) {
                synth.cancel(); // removes all utterances from the utterance queue.
                setIsPlaying(false);
                return;
            }
            const utterance: SpeechSynthesisUtterance | undefined = getUtterance(answer);

            if (!utterance) {
                return;
            }

            synth.speak(utterance);

            utterance.onstart = () => {
                setIsPlaying(true);
                return;
            };

            utterance.onend = () => {
                setIsPlaying(false);
                return;
            };
        }
    };
    const color = isPlaying ? "red" : "black";

    return (
        <IconButton
            style={{ color: color }}
            iconProps={{ iconName: "Volume3" }}
            title="音声返答"
            ariaLabel="音声返答"
            onClick={() => startOrStopSpeech(answer)}
            disabled={!synth}
        />
    );
};
