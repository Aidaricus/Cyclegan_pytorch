import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats


def calculate_accuracy(predictions, targets):
    """Вычисление точности классификации"""
    if isinstance(predictions, torch.Tensor):
        predictions = torch.argmax(predictions, dim=1)

    if isinstance(targets, torch.Tensor):
        if targets.dim() > 1:  # one-hot encoding
            targets = torch.argmax(targets, dim=1)

        return (predictions == targets).float().mean().item()
    else:
        return np.mean(predictions == targets)


def calculate_confusion_matrix(predictions, targets):
    """Вычисление матрицы ошибок"""
    return confusion_matrix(targets, predictions)


def calculate_classification_report(predictions, targets, class_names=None):
    """Детальный отчет по классификации"""
    return classification_report(targets, predictions, target_names=class_names)


def calculate_transfer_strength(source_prob, transfer_prob, cycle_prob):
    """
    Вычисление силы стилевого переноса

    Args:
        source_prob: вероятность исходного класса для оригинала
        transfer_prob: вероятность исходного класса для перенесенного
        cycle_prob: вероятность исходного класса для цикла

    Returns:
        transfer_strength: сила переноса [0, 1]
    """
    # Формула из статьи: (P(A|x_A) - P(A|x_B) + P(A|x_A_) - P(A|x_B)) / 2
    strength = (source_prob - transfer_prob + cycle_prob - transfer_prob) / 2.0
    return max(0.0, min(1.0, strength))  # Ограничиваем [0, 1]


def calculate_content_preservation(original, transferred):
    """
    Вычисление меры сохранения контента (MSE между оригиналом и переносом)

    Args:
        original: оригинальная матрица
        transferred: перенесенная матрица

    Returns:
        content_diff: разность контента (MSE)
    """
    return np.mean((original.astype(np.float32) - transferred.astype(np.float32)) ** 2)


def calculate_transfer_success_rate(source_probs, transfer_probs, threshold=0.5):
    """
    Вычисление процента успешных переносов

    Args:
        source_probs: вероятности исходного класса для оригиналов
        transfer_probs: вероятности исходного класса для переносов
        threshold: пороговое значение

    Returns:
        success_rate: процент успешных переносов
    """
    successful = 0
    total = len(source_probs)

    for source_p, transfer_p in zip(source_probs, transfer_probs):
        if source_p > threshold and transfer_p < threshold:
            successful += 1

    return successful / total if total > 0 else 0.0


def calculate_genre_transfer_metrics(results, which_direction='AtoB'):
    """
    Вычисление комплексных метрик переноса жанра

    Args:
        results: список результатов оценки
        which_direction: направление переноса

    Returns:
        metrics: словарь с метриками
    """
    successful_results = [r for r in results if r['is_successful']]

    metrics = {
        'total_samples': len(results),
        'successful_transfers': len(successful_results),
        'success_rate': len(successful_results) / len(results) if results else 0.0,
        'avg_transfer_strength': np.mean(
            [r['transfer_strength'] for r in successful_results]) if successful_results else 0.0,
        'std_transfer_strength': np.std(
            [r['transfer_strength'] for r in successful_results]) if successful_results else 0.0,
        'avg_content_preservation': np.mean([r['content_diff'] for r in results]),
        'avg_prob_difference': np.mean([r['prob_diff'] for r in results]),
        'median_prob_difference': np.median([r['prob_diff'] for r in results]),
    }

    # Дополнительная статистика для успешных переносов
    if successful_results:
        transfer_strengths = [r['transfer_strength'] for r in successful_results]
        metrics.update({
            'min_transfer_strength': np.min(transfer_strengths),
            'max_transfer_strength': np.max(transfer_strengths),
            'q25_transfer_strength': np.percentile(transfer_strengths, 25),
            'q75_transfer_strength': np.percentile(transfer_strengths, 75),
        })

    return metrics


def calculate_classifier_robustness_metrics(accuracy_results):
    """
    Метрики робастности классификатора к шуму

    Args:
        accuracy_results: словарь {noise_level: accuracy}

    Returns:
        robustness_metrics: метрики робастности
    """
    noise_levels = sorted(accuracy_results.keys())
    accuracies = [accuracy_results[level] for level in noise_levels]

    # Базовая точность (без шума)
    baseline_accuracy = accuracy_results[0.0]

    # Деградация точности
    accuracy_drops = [baseline_accuracy - acc for acc in accuracies]

    # Площадь под кривой (чем больше, тем хуже робастность)
    auc_degradation = np.trapz(accuracy_drops, noise_levels)

    metrics = {
        'baseline_accuracy': baseline_accuracy,
        'accuracy_at_noise_levels': accuracy_results,
        'max_accuracy_drop': max(accuracy_drops),
        'auc_degradation': auc_degradation,
        'robustness_score': 1.0 - (auc_degradation / (noise_levels[-1] * baseline_accuracy))
        # Нормализованная робастность
    }

    return metrics


def statistical_significance_test(group1_scores, group2_scores, test='ttest'):
    """
    Тест статистической значимости различий между группами

    Args:
        group1_scores: оценки первой группы
        group2_scores: оценки второй группы
        test: тип теста ('ttest', 'mannwhitney')

    Returns:
        test_result: результат теста
    """
    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(group1_scores, group2_scores)
        test_name = "Independent t-test"
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': calculate_effect_size(group1_scores, group2_scores)
    }


def calculate_effect_size(group1, group2):
    """Вычисление размера эффекта (Cohen's d)"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    return d


def generate_ranking_report(results, save_path, which_direction='AtoB'):
    """
    Генерация текстового отчета с рейтингом

    Args:
        results: результаты оценки
        save_path: путь для сохранения
        which_direction: направление переноса
    """
    # Сортировка по разности вероятностей
    sorted_results = sorted(results, key=lambda x: x['prob_diff'], reverse=True)

    with open(save_path, 'w') as f:
        f.write(f"Genre Transfer Evaluation Report - {which_direction}\n")
        f.write("=" * 80 + "\n\n")

        # Общая статистика
        metrics = calculate_genre_transfer_metrics(results, which_direction)
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {metrics['total_samples']}\n")
        f.write(f"Successful transfers: {metrics['successful_transfers']}\n")
        f.write(f"Success rate: {metrics['success_rate']:.4f}\n")
        f.write(f"Average transfer strength: {metrics['avg_transfer_strength']:.4f}\n")
        f.write(f"Average content preservation (MSE): {metrics['avg_content_preservation']:.6f}\n")
        f.write(f"Average probability difference: {metrics['avg_prob_difference']:.4f}\n\n")

        # Детальная таблица
        f.write("DETAILED RESULTS (sorted by probability difference):\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'ID':>3} {'Content_MSE':>12} {'Prob_Diff':>10} {'Source_P':>9} {'Transfer_P':>11} {'Cycle_P':>8} {'Strength':>9} {'Success':>8}\n")
        f.write("-" * 80 + "\n")

        for result in sorted_results:
            f.write(f"{result['id']:>3} "
                    f"{result['content_diff']:>12.6f} "
                    f"{result['prob_diff']:>10.6f} "
                    f"{result['source_prob']:>9.6f} "
                    f"{result['transfer_prob']:>11.6f} "
                    f"{result['cycle_prob']:>8.6f} "
                    f"{result['transfer_strength']:>9.6f} "
                    f"{'Yes' if result['is_successful'] else 'No':>8}\n")

    print(f"Ranking report saved: {save_path}")
