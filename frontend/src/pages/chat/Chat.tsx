import logoImage from "../../assets/FeaturedDefault.png";
import { useRef, useState, useEffect } from "react";
import { Checkbox, Panel, DefaultButton, TextField, ITextFieldProps, ICheckboxProps } from "@fluentui/react";
import { useId } from "@fluentui/react-hooks";

import styles from "./Chat.module.css";

import { RetrievalMode, ChatCompletionResponse, ChatCompletionDeltaResponse, ChatAppRequestOptions } from "../../api";
import { AIChatProtocolClient, AIChatMessage } from "@microsoft/ai-chat-protocol";
import { Answer, AnswerError, AnswerLoading } from "../../components/Answer";
import { QuestionInput } from "../../components/QuestionInput";
import { ExampleList } from "../../components/Example";
import { UserChatMessage } from "../../components/UserChatMessage";
import { HelpCallout } from "../../components/HelpCallout";
import { AnalysisPanel, AnalysisPanelTabs } from "../../components/AnalysisPanel";
import { SettingsButton } from "../../components/SettingsButton";
import { ClearChatButton } from "../../components/ClearChatButton";
import { VectorSettings } from "../../components/VectorSettings";
import { toolTipText } from "../../i18n/tooltips.js";

const Chat = () => {
    const chatInputTextFieldPlaceholder: string = `新しい質問を入力してください（例：${import.meta.env.VITE_CHAT_EXAMPLE_1}）`;

    const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false);
    const [temperature, setTemperature] = useState<number>(0.3);
    const [retrieveCount, setRetrieveCount] = useState<number>(3);
    const [scoreThreshold, setScoreThreshold] = useState<number>(0.5);
    const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(RetrievalMode.Hybrid);

    const lastQuestionRef = useRef<string>("");
    const chatMessageStreamEnd = useRef<HTMLDivElement | null>(null);

    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [isStreaming, setIsStreaming] = useState<boolean>(false);
    const [shouldStream, setShouldStream] = useState<boolean>(false);
    const [error, setError] = useState<unknown>();

    const [activeAnalysisPanelTab, setActiveAnalysisPanelTab] = useState<AnalysisPanelTabs | undefined>(undefined);

    const [selectedAnswer, setSelectedAnswer] = useState<number>(0);
    const [answers, setAnswers] = useState<[user: string, response: ChatCompletionResponse][]>([]);
    const [sessionState, setSessionState] = useState<object | null>(null);

    const [streamedAnswers, setStreamedAnswers] = useState<[user: string, response: ChatCompletionResponse][]>([]);

    const handleAsyncResponse = async (question: string, answers: [string, ChatCompletionResponse][], result: AsyncIterable<ChatCompletionDeltaResponse>) => {
        let answer = "";
        const chatCompletion: ChatCompletionResponse = {
            context: {
                data_points: [],
                thoughts: []
            },
            message: { content: "", role: "assistant" }
        };
        const updateState = (newContent: string) => {
            return new Promise(resolve => {
                setTimeout(() => {
                    answer += newContent;
                    // We need to create a new object to trigger a re-render
                    const latestCompletion: ChatCompletionResponse = {
                        ...chatCompletion,
                        message: { content: answer, role: chatCompletion.message.role }
                    };
                    setStreamedAnswers([...answers, [question, latestCompletion]]);
                    resolve(null);
                }, 33);
            });
        };
        try {
            setIsStreaming(true);
            for await (const response of result) {
                if (response.context) {
                    chatCompletion.context = {
                        ...chatCompletion.context,
                        ...response.context
                    };
                }
                if (response.delta && response.delta.role) {
                    chatCompletion.message.role = response.delta.role;
                }
                if (response.delta && response.delta.content) {
                    setIsLoading(false);
                    await updateState(response.delta.content);
                }
            }
        } finally {
            setIsStreaming(false);
        }
        chatCompletion.message.content = answer;
        return chatCompletion;
    };
    const makeApiRequest = async (question: string) => {
        lastQuestionRef.current = question;

        error && setError(undefined);
        setIsLoading(true);
        setActiveAnalysisPanelTab(undefined);
        try {
            const messages: AIChatMessage[] = answers.flatMap(a => [
                { content: a[0], role: "user" },
                { content: a[1].message.content, role: "assistant" }
            ]);

            const allMessages: AIChatMessage[] = [...messages, { content: question, role: "user" }];
            const options: ChatAppRequestOptions = {
                context: {
                    overrides: {
                        top: retrieveCount,
                        retrieval_mode: retrievalMode,
                        temperature: temperature,
                        score_threshold: scoreThreshold
                    }
                },
                sessionState: sessionState ? sessionState : null
            };
            const chatClient: AIChatProtocolClient = new AIChatProtocolClient("/chat");
            if (shouldStream) {
                const result = (await chatClient.getStreamedCompletion(allMessages, options)) as AsyncIterable<ChatCompletionDeltaResponse>;
                const parsedResponse = await handleAsyncResponse(question, answers, result);
                setAnswers([...answers, [question, parsedResponse]]);
                setSessionState(parsedResponse?.sessionState ? parsedResponse.sessionState : null);
            } else {
                const result = (await chatClient.getCompletion(allMessages, options)) as ChatCompletionResponse;
                setAnswers([...answers, [question, result]]);
                setSessionState(result?.sessionState ? result.sessionState : null);
            }
        } catch (e) {
            setError(e);
        } finally {
            setIsLoading(false);
        }
    };

    const checkThenMakeApiRequest = async (question: string) => {
        lastQuestionRef.current = question;

        makeApiRequest(question);
    };

    const clearChat = () => {
        lastQuestionRef.current = "";
        error && setError(undefined);
        setActiveAnalysisPanelTab(undefined);
        setAnswers([]);
        setStreamedAnswers([]);
        setIsLoading(false);
        setIsStreaming(false);
    };

    useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "smooth" }), [isLoading]);
    useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "auto" }), [streamedAnswers]);

    const onTemperatureChange = (_ev?: React.SyntheticEvent<HTMLElement, Event>, newValue?: string) => {
        setTemperature(parseFloat(newValue || "0"));
    };

    const onScoreThresholdChange = (_ev?: React.SyntheticEvent<HTMLElement, Event>, newValue?: string) => {
        setScoreThreshold(parseFloat(newValue || "0"));
    };

    const onRetrieveCountChange = (_ev?: React.SyntheticEvent<HTMLElement, Event>, newValue?: string) => {
        setRetrieveCount(parseInt(newValue || "3"));
    };

    const onShouldStreamChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setShouldStream(!!checked);
    };

    const onExampleClicked = (example: string) => {
        checkThenMakeApiRequest(example);
    };

    const onToggleTab = (tab: AnalysisPanelTabs, index: number) => {
        if (activeAnalysisPanelTab === tab && selectedAnswer === index) {
            setActiveAnalysisPanelTab(undefined);
        } else {
            setActiveAnalysisPanelTab(tab);
        }

        setSelectedAnswer(index);
    };

    // IDs for form labels and their associated callouts
    const temperatureId = useId("temperature");
    const temperatureFieldId = useId("temperatureField");
    const searchScoreId = useId("searchScore");
    const searchScoreFieldId = useId("searchScoreField");
    const retrieveCountId = useId("retrieveCount");
    const retrieveCountFieldId = useId("retrieveCountField");
    const shouldStreamId = useId("shouldStream");
    const shouldStreamFieldId = useId("shouldStreamField");

    return (
        <div className={styles.container}>
            <div className={styles.commandsContainer}>
                <ClearChatButton className={styles.commandButton} onClick={clearChat} disabled={!lastQuestionRef.current || isLoading} />
                <SettingsButton className={styles.commandButton} onClick={() => setIsConfigPanelOpen(!isConfigPanelOpen)} />
            </div>
            <div className={styles.chatRoot}>
                <div className={styles.chatContainer}>
                    {!lastQuestionRef.current ? (
                        <div className={styles.chatEmptyState}>
                            <img src={logoImage} alt="App logo" aria-label="App logo" width="100px" height="100px" className={styles.githubLogo} />
                            <h1 className={styles.chatEmptyStateTitle}>{import.meta.env.VITE_APP_HEADING}</h1>
                            <h2 className={styles.chatEmptyStateSubtitle}>何でも質問してください</h2>
                            <ExampleList onExampleClicked={onExampleClicked} />
                        </div>
                    ) : (
                        <div className={styles.chatMessageStream}>
                            {isStreaming &&
                                streamedAnswers.map((streamedAnswer, index) => (
                                    <div key={index}>
                                        <UserChatMessage message={streamedAnswer[0]} />
                                        <div className={styles.chatMessageGpt}>
                                            <Answer
                                                isStreaming={true}
                                                key={index}
                                                answer={streamedAnswer[1]}
                                                isSelected={false}
                                                onThoughtProcessClicked={() => onToggleTab(AnalysisPanelTabs.ThoughtProcessTab, index)}
                                                onSupportingContentClicked={() => onToggleTab(AnalysisPanelTabs.SupportingContentTab, index)}
                                            />
                                        </div>
                                    </div>
                                ))}
                            {!isStreaming &&
                                answers.map((answer, index) => (
                                    <div key={index}>
                                        <UserChatMessage message={answer[0]} />
                                        <div className={styles.chatMessageGpt}>
                                            <Answer
                                                isStreaming={false}
                                                key={index}
                                                answer={answer[1]}
                                                isSelected={selectedAnswer === index && activeAnalysisPanelTab !== undefined}
                                                onThoughtProcessClicked={() => onToggleTab(AnalysisPanelTabs.ThoughtProcessTab, index)}
                                                onSupportingContentClicked={() => onToggleTab(AnalysisPanelTabs.SupportingContentTab, index)}
                                            />
                                        </div>
                                    </div>
                                ))}
                            {isLoading && (
                                <>
                                    <UserChatMessage message={lastQuestionRef.current} />
                                    <div className={styles.chatMessageGptMinWidth}>
                                        <AnswerLoading />
                                    </div>
                                </>
                            )}
                            {error ? (
                                <>
                                    <UserChatMessage message={lastQuestionRef.current} />
                                    <div className={styles.chatMessageGptMinWidth}>
                                        <AnswerError error={error.toString()} onRetry={() => checkThenMakeApiRequest(lastQuestionRef.current)} />
                                    </div>
                                </>
                            ) : null}
                            <div ref={chatMessageStreamEnd} />
                        </div>
                    )}

                    <div className={styles.chatInput}>
                        <QuestionInput
                            clearOnSend
                            placeholder={chatInputTextFieldPlaceholder}
                            disabled={isLoading}
                            onSend={question => checkThenMakeApiRequest(question)}
                        />
                    </div>
                </div>

                {answers.length > 0 && activeAnalysisPanelTab && (
                    <AnalysisPanel
                        className={styles.chatAnalysisPanel}
                        onActiveTabChanged={x => onToggleTab(x, selectedAnswer)}
                        answer={answers[selectedAnswer][1]}
                        activeTab={activeAnalysisPanelTab}
                    />
                )}

                <Panel
                    headerText="回答生成の設定"
                    isOpen={isConfigPanelOpen}
                    isBlocking={false}
                    onDismiss={() => setIsConfigPanelOpen(false)}
                    closeButtonAriaLabel="Close"
                    onRenderFooterContent={() => <DefaultButton onClick={() => setIsConfigPanelOpen(false)}>閉じる</DefaultButton>}
                    isFooterAtBottom={true}
                >
                    <TextField
                        id={temperatureFieldId}
                        className={styles.chatSettingsSeparator}
                        label="創造性"
                        type="number"
                        min={0}
                        max={1}
                        step={0.1}
                        defaultValue={temperature.toString()}
                        onChange={onTemperatureChange}
                        aria-labelledby={temperatureId}
                        onRenderLabel={(props: ITextFieldProps | undefined) => (
                            <HelpCallout labelId={temperatureId} fieldId={temperatureFieldId} helpText={toolTipText.temperature} label={props?.label} />
                        )}
                    />

                    <TextField
                        id={searchScoreFieldId}
                        className={styles.chatSettingsSeparator}
                        label="類似度スコアの閾値"
                        type="number"
                        min={0}
                        max={1}
                        step={0.1}
                        defaultValue={scoreThreshold.toString()}
                        onChange={onScoreThresholdChange}
                        aria-labelledby={searchScoreId}
                        onRenderLabel={(props: ITextFieldProps | undefined) => (
                            <HelpCallout labelId={searchScoreId} fieldId={searchScoreFieldId} helpText={toolTipText.searchScore} label={props?.label} />
                        )}
                    />

                    <TextField
                        id={retrieveCountFieldId}
                        className={styles.chatSettingsSeparator}
                        label="検索結果の取得数:"
                        type="number"
                        min={1}
                        max={20}
                        defaultValue={retrieveCount.toString()}
                        onChange={onRetrieveCountChange}
                        aria-labelledby={retrieveCountId}
                        onRenderLabel={(props: ITextFieldProps | undefined) => (
                            <HelpCallout labelId={retrieveCountId} fieldId={retrieveCountFieldId} helpText={toolTipText.retrieveNumber} label={props?.label} />
                        )}
                    />

                    <VectorSettings
                        defaultRetrievalMode={retrievalMode}
                        updateRetrievalMode={(retrievalMode: RetrievalMode) => setRetrievalMode(retrievalMode)}
                    />

                    <Checkbox
                        id={shouldStreamFieldId}
                        className={styles.chatSettingsSeparator}
                        checked={shouldStream}
                        label="チャットの完了応答をストリーミングする"
                        onChange={onShouldStreamChange}
                        aria-labelledby={shouldStreamId}
                        onRenderLabel={(props: ICheckboxProps | undefined) => (
                            <HelpCallout labelId={shouldStreamId} fieldId={shouldStreamFieldId} helpText={toolTipText.streamChat} label={props?.label} />
                        )}
                    />
                </Panel>
            </div>
        </div>
    );
};

export default Chat;
