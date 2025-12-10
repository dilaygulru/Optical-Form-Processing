#include "AnswerKey.hpp"
#include <sstream>
#include <algorithm>

AnswerKey::AnswerKey() {
    // Constructor - boş başlatılır, loadAnswerKey ile doldurulur
}

void AnswerKey::loadAnswerKey(const std::vector<QuestionAnswer>& answers) {
    answerKey_.clear();
    
    for (const auto& qa : answers) {
        answerKey_[qa.subject][qa.questionNumber] = qa.correctAnswer;
    }
}

std::vector<char> AnswerKey::parseAnswerString(const std::string& answerStr) {
    std::vector<char> result;
    std::stringstream ss(answerStr);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        if (token.empty() || token == "-") {
            result.push_back('-');
        } else {
            result.push_back(token[0]);
        }
    }
    
    return result;
}

AnswerKey::ScoreResult AnswerKey::calculateScore(
    const std::map<std::string, std::string>& studentAnswers) {
    
    ScoreResult result;
    result.totalQuestions = 0;
    result.correctAnswers = 0;
    result.wrongAnswers = 0;
    result.emptyAnswers = 0;
    result.score = 0.0;
    
    // Her ders için kontrol et
    for (const auto& subjectPair : answerKey_) {
        const std::string& subject = subjectPair.first;
        const auto& correctAnswers = subjectPair.second;
        
        // Öğrencinin bu derse ait cevaplarını al
        auto it = studentAnswers.find(subject);
        if (it == studentAnswers.end()) {
            // Öğrenci bu dersi işaretlememiş
            result.emptyAnswers += correctAnswers.size();
            result.totalQuestions += correctAnswers.size();
            continue;
        }
        
        std::vector<char> studentAns = parseAnswerString(it->second);
        
        int subjectCorrect = 0;
        
        // Her soruyu kontrol et
        for (const auto& qaPair : correctAnswers) {
            int qIndex = qaPair.first;
            char correctAns = qaPair.second;
            
            result.totalQuestions++;
            
            // Öğrenci cevabını al
            char studentChar = '-';
            if (qIndex < (int)studentAns.size()) {
                studentChar = studentAns[qIndex];
            }
            
            if (studentChar == '-') {
                result.emptyAnswers++;
            } else if (studentChar == correctAns) {
                result.correctAnswers++;
                subjectCorrect++;
            } else {
                result.wrongAnswers++;
            }
        }
        
        result.subjectScores[subject] = subjectCorrect;
    }
    
    // Puanı hesapla (100 üzerinden)
    if (result.totalQuestions > 0) {
        result.score = (double)result.correctAnswers * 100.0 / result.totalQuestions;
    }
    
    return result;
}

bool AnswerKey::isCorrect(const std::string& subject, int questionIndex, char answer) {
    auto subjectIt = answerKey_.find(subject);
    if (subjectIt == answerKey_.end()) {
        return false;
    }
    
    auto ansIt = subjectIt->second.find(questionIndex);
    if (ansIt == subjectIt->second.end()) {
        return false;
    }
    
    return ansIt->second == answer;
}

char AnswerKey::getCorrectAnswer(const std::string& subject, int questionIndex) {
    auto subjectIt = answerKey_.find(subject);
    if (subjectIt == answerKey_.end()) {
        return '-';
    }
    
    auto ansIt = subjectIt->second.find(questionIndex);
    if (ansIt == subjectIt->second.end()) {
        return '-';
    }
    
    return ansIt->second;
}