import type {
	DeepLTranslateResult,
	LibreTranslateResult,
	TranslateData,
	TranslateParams,
	TranslationTypeOption,
} from "@/types/servies/translation";
import { withCache } from "@/utils/cache";
import { ServiceResponse, serviceBaseFetch, serviceFetch } from ".";

export const translate = async (params: TranslateParams) => {
	return serviceFetch<TranslateData>("/api/v2/translation/translate", {
		method: "POST",
		data: params,
	});
};

export const getTranslationTypes = async () => {
	return serviceFetch<TranslationTypeOption[]>("/api/v2/translation/types", {
		method: "GET",
	});
};

const fetchTranslationTypes = async (): Promise<
	TranslationTypeOption[] | undefined
> => {
	const resp = await getTranslationTypes();
	if (resp.success()) {
		return resp.data ?? [];
	}
	return undefined;
};

export const getTranslationTypesWithCache = withCache(fetchTranslationTypes, {
	key: "getTranslationTypes",
	duration: 60 * 60 * 1000, // 缓存 1 小时
});

export const translateTextDeepL = async (
	apiUri: string,
	apiKey: string,
	sourceContent: string[],
	sourceLanguage: string | null,
	targetLanguage: string,
	preferQualityOptimized: boolean,
): Promise<DeepLTranslateResult | undefined> => {
	const response = await serviceBaseFetch(apiUri, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `DeepL-Auth-Key ${apiKey}`,
		},
		data: {
			text: sourceContent,
			source_lang: sourceLanguage,
			target_lang: targetLanguage,
			preserve_formatting: true,
			model_type: preferQualityOptimized
				? "prefer_quality_optimized"
				: "latency_optimized",
		},
	});

	if (response instanceof ServiceResponse) {
		response.success();
		return undefined;
	}

	return (await response.json()) as DeepLTranslateResult;
};

export const translateTextLibreTranslate = async (
	apiUri: string,
	apiKey: string,
	sourceContent: string[],
	sourceLanguage: string | null,
	targetLanguage: string,
): Promise<LibreTranslateResult | undefined> => {
	const response = await serviceBaseFetch(apiUri, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
		},
		data: {
			q: sourceContent,
			source: sourceLanguage ?? "auto",
			target: targetLanguage,
			format: "text",
		},
	});

	if (response instanceof ServiceResponse) {
		response.success();
		return undefined;
	}

	return (await response.json()) as LibreTranslateResult;
};

export const translateTextOpenAiCompatible = async (
	apiUri: string,
	apiKey: string,
	apiModel: string,
	sourceContent: string[],
	systemPrompt: string,
	maxTokens: number,
	temperature: number,
): Promise<string | undefined> => {
	const body: Record<string, unknown> = {
		model: apiModel,
		messages: [
			{ role: "system", content: systemPrompt },
			{ role: "user", content: sourceContent.join("%%") },
		],
		max_completion_tokens: maxTokens,
		temperature,
		stream: false,
	};

	const response = await serviceBaseFetch(
		`${apiUri.endsWith("/") ? apiUri : `${apiUri}/`}chat/completions`,
		{
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
			},
			data: body,
		},
	);

	if (response instanceof ServiceResponse) {
		response.success();
		return undefined;
	}

	const data = (await response.json()) as {
		choices?: { message?: { content?: string } }[];
	};

	return data.choices?.[0]?.message?.content;
};
