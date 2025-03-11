import { SetStateAction, useState } from "react";
import { Button, Tooltip } from "@fluentui/react-components";
import { Mic28Filled } from "@fluentui/react-icons";
import styles from "./QuestionInput.module.css";

interface Props {
    updateQuestion: (question: string) => void;
}

const SpeechRecognition = (window as any).speechRecognition || (window as any).webkitSpeechRecognition;
let speechRecognition: {
    continuous: boolean;
    lang: string;
    interimResults: boolean;
    maxAlternatives: number;
    start: () => void;
    onresult: (event: { results: { transcript: SetStateAction<string> }[][] }) => void;
    onend: () => void;
    onerror: (event: { error: string }) => void;
    stop: () => void;
} | null = null;
try {
    speechRecognition = new SpeechRecognition();
    if (speechRecognition != null) {
        speechRecognition.lang = "ja-JP";
        speechRecognition.interimResults = true;
    }
} catch (err) {
    console.error("音声認識はサポートされていません");
    speechRecognition = null;
}

export const SpeechInput = ({ updateQuestion }: Props) => {
    const [isRecording, setIsRecording] = useState<boolean>(false);

    const startRecording = () => {
        if (speechRecognition == null) {
            console.error("音声認識はサポートされていません");
            return;
        }

        speechRecognition.onresult = (event: { results: { transcript: SetStateAction<string> }[][] }) => {
            let input = "";
            for (const result of event.results) {
                input += result[0].transcript;
            }
            updateQuestion(input);
        };
        speechRecognition.onend = () => {
            // NOTE: In some browsers (e.g. Chrome), the recording will stop automatically after a few seconds of silence.
            setIsRecording(false);
        };
        speechRecognition.onerror = (event: { error: string }) => {
            if (speechRecognition) {
                speechRecognition.stop();
                if (event.error == "no-speech") {
                    alert("音声が検出されませんでした。システムのオーディオ設定を確認して、もう一度お試しください。");
                } else if (event.error == "language-not-supported") {
                    alert(
                        `音声認識エラーが検出されました: ${event.error}. 音声認識入力機能は、すべてのブラウザでまだ動作しません。例えば、ARMチップを搭載したMac OS XのEdgeでは動作しません。別のブラウザやOSを試してください。`
                    );
                } else {
                    alert(`音声認識エラーが検出されました: ${event.error}.`);
                }
            }
        };

        setIsRecording(true);
        speechRecognition.start();
    };

    const stopRecording = () => {
        if (speechRecognition == null) {
            console.error("音声認識はサポートされていません");
            return;
        }
        speechRecognition.stop();
        setIsRecording(false);
    };

    if (speechRecognition == null) {
        return <></>;
    }
    return (
        <>
            {!isRecording && (
                <div className={styles.questionInputButtonsContainer}>
                    <Tooltip content="音声での質問を開始する" relationship="label">
                        <Button size="large" icon={<Mic28Filled primaryFill="rgba(115, 118, 225, 1)" />} onClick={startRecording} />
                    </Tooltip>
                </div>
            )}
            {isRecording && (
                <div className={styles.questionInputButtonsContainer}>
                    <Tooltip content="音声での質問を停止する" relationship="label">
                        <Button size="large" icon={<Mic28Filled primaryFill="rgba(250, 0, 0, 0.7)" />} disabled={!isRecording} onClick={stopRecording} />
                    </Tooltip>
                </div>
            )}
        </>
    );
};
