import OpenAI from "openai";
import "https://deno.land/std@0.214.0/dotenv/load.ts";
import { parse, stringify } from "jsr:@std/csv";

interface LLMConfig {
  apiEndpoint: string;
  apiKey: string;
  modelName: string;
}

// 評価者のLLM設定
const evaluatorConfig: LLMConfig = {
  apiEndpoint: Deno.env.get("EVALUATOR_API_ENDPOINT") ?? "https://api.openai.com/v1",
  apiKey: Deno.env.get("EVALUATOR_API_KEY") ?? "",
  modelName: Deno.env.get("EVALUATOR_MODEL_NAME") ?? "gpt-4o",
};

// ベンチ対象のLLM設定
const targetConfig: LLMConfig = {
  apiEndpoint: Deno.env.get("TARGET_API_ENDPOINT") ?? "https://api.openai.com/v1",
  apiKey: Deno.env.get("TARGET_API_KEY") ?? "",
  modelName: Deno.env.get("TARGET_MODEL_NAME") ?? "",
};

// OpenAIクライアントを作成
function createOpenAIClient(config: LLMConfig): OpenAI {
  return new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.apiEndpoint,
  });
}

const evaluatorClient = createOpenAIClient(evaluatorConfig);
const targetClient = createOpenAIClient(targetConfig);

async function generate_answer(input: string): Promise<string> {
	const response = await targetClient.chat.completions.create({
		model: targetConfig.modelName,
		messages: [{ role: "user", content: input }],
	});
	return response.choices[0].message.content ?? "";
}

async function evaluate_answer(input: string, answer: string, example_output: string, eval_aspect: string): Promise<number> {
	const prompt = `問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

# 指示
「採点基準」と「正解例」を参考にして、、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{input_text}

# 正解例
{output_text}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 言語モデルの回答
{pred}

# ここまでが'言語モデルの回答'です。回答が空白だった場合、1点にしてください。

# 指示
「採点基準」と「正解例」を参考にして、、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。`
	
	const response = await evaluatorClient.chat.completions.create({
		model: evaluatorConfig.modelName,
		messages: [{ role: "user", content: prompt.replace("{input_text}", input)
			.replace("{output_text}", example_output)
			.replace("{eval_aspect}", eval_aspect)
			.replace("{pred}", answer) }],
	});
	return Number(response.choices[0].message.content);
}


// デバッグ用のログ出力
console.log("環境変数の確認:");
console.log("EVALUATOR_API_ENDPOINT:", Deno.env.get("EVALUATOR_API_ENDPOINT"));
console.log("EVALUATOR_API_KEY:", Deno.env.get("EVALUATOR_API_KEY")?.slice(0, 10) + "...");
console.log("EVALUATOR_MODEL_NAME:", Deno.env.get("EVALUATOR_MODEL_NAME"));

const elyza_csv = await Deno.readTextFile("./test.csv");
let elyza_data = parse(elyza_csv, {
    skipFirstRow: true,
	strip: true,
});

//面倒なので最初の10個以外は切り捨てる
//elyza_data = elyza_data.slice(0, 10);


let results = [];

// モデルの回答を生成（並列処理）
console.log("評価対象モデルの回答を生成中...");
const generatePromises = elyza_data.map(async (data, i) => {
    const answer = await generate_answer(data.input);
    console.log(`回答生成完了: ${i + 1}/${elyza_data.length}`);
    return {
        input: data.input,
        output: answer,
        example_output: data.output,
        eval_aspect: data.eval_aspect,
        score: -1,
    };
});

// 5件ずつ並列処理
for (let i = 0; i < elyza_data.length; i += 5) {
    const chunk = generatePromises.slice(i, i + 5);
    const chunkResults = await Promise.all(chunk);
    results.push(...chunkResults);
}
console.log("評価対象モデルの回答生成が完了しました");

// 評価対象モデルの回答を評価（並列処理）
console.log("評価対象モデルの回答を評価中...");
const evaluatePromises = results.map(async (data, i) => {
    const score = await evaluate_answer(data.input, data.output, data.example_output, data.eval_aspect);
    console.log(`回答評価完了: ${i + 1}/${results.length}`);
    return score;
});

// 5件ずつ並列処理
for (let i = 0; i < results.length; i += 5) {
    const chunk = evaluatePromises.slice(i, i + 5);
    const scores = await Promise.all(chunk);
    scores.forEach((score, index) => {
        results[i + index].score = score;
    });
}
console.log("評価対象モデルの回答評価が完了しました");

//平均点算出
const average_score = results.reduce((sum, data) => sum + data.score, 0) / results.length;
console.log("平均点: " + average_score);

let output_json = {
	model_name: targetConfig.modelName,
	average_score: average_score,
	results: results,
}

// resultsをJSONファイルとして保存する部分を修正
await Deno.writeTextFile(
    `results.json`, 
    JSON.stringify(output_json, null, 4)
);

