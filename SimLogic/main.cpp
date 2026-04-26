#include "sim.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <random>

#define CSV_FILE "SimOut.csv"

std::mt19937 rng(std::random_device{}());
std::normal_distribution<float> noise(0.0f, 100.0f);

/**
 * @def Random generates a spawn point for the threats
 *      The points needs to be within a predefined grid SP_X1, SP_X2, etc
 */
Vector2 GSP()
{
    std::uniform_real_distribution<float> distX(SP_X1, SP_X2);
    std::uniform_real_distribution<float> distY(SP_Y1, SP_Y2);

    return {distX(rng), distY(rng)};
}

float kmh_to_ms(float kmh)
{ return kmh / 3.6f; }

float random_float(float minVal, float maxVal)
{
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    return dist(rng);
}

float apply_initial_heading_offset(Vector2 Tpos, Vector2 TargetPos, float maxOffset)
{
    float baseHeading = angle_to_target(Tpos, TargetPos);
    float offset = random_float(-maxOffset, maxOffset);

    return baseHeading + offset;
}

/**
 * @def Random generates threats
 * @param id individual unique id
 * @param targets vector containing all existing possible targets
 */
Threat get_threat(unsigned int id, const std::vector<Target>& targets)
{
    std::uniform_int_distribution<int> typeDist(0, 2);
    std::uniform_int_distribution<int> targetDist(0, static_cast<int>(targets.size()) - 1);

    int type = typeDist(rng);
    Target target = targets[targetDist(rng)];

    Vector2 spawn = GSP();

    if (type == KALIBR)
    {
        float heading = apply_initial_heading_offset(spawn, target.position, 1.2f);
        Threat T(id, spawn, target, heading, 0.75f, kmh_to_ms(3087.0f), "Kalibr", KALIBR, 1.2f, 0.45f);
        return T;
    }

    if (type == KINZHAL)
    {
        float heading = apply_initial_heading_offset(spawn, target.position, 0.50f);
        Threat T(id, spawn, target, heading, 1.0f, kmh_to_ms(4500.0f), "Kinzhal", KINZHAL, 2.4f, 0.12f);
        return T;
    }

    float heading = apply_initial_heading_offset(spawn, target.position, 0.80f);
    Threat T(id, spawn, target, heading, 0.5f, kmh_to_ms(1500.0f), "Geran2", GERAN, 2.8f, 1.10f);

    return T;
}

const Target* findTargetInfo(const std::vector<Target>& targets, Vector2 targetPosition)
{
    for (const Target& target : targets)
        if (target.position.x == targetPosition.x && target.position.y == targetPosition.y)
            return &target;

    return nullptr;
}

int getRiskLevel(int threatType, int targetType)
{
    int riskMatrix[3][3] =
    {
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 4}
    };

    return riskMatrix[threatType][targetType];
}

float normalize_speed(float speed)
{
    float speed_n = (speed - kmh_to_ms(1500.0f)) / (kmh_to_ms(5000.0f) - kmh_to_ms(1500.0f));

    if (speed_n < 0.0f) speed_n = 0.0f;
    if (speed_n > 1.0f) speed_n = 1.0f;

    return speed_n;
}

void run_simulation(float dt, const unsigned int steps, std::vector<Threat> &threats, std::vector<Target> &targets)
{
    std::ofstream fout(CSV_FILE);

    fout << "EnemyWeaponType,ID,UAV_ID,TrueX,TrueY,MeasuredX,MeasuredY,MeasuredX_Normalized,MeasuredY_Normalized,"
         << "Vx,Vy,Tx,Ty,Speed,Speed_Normalized,Heading,TurnGain,MaxTurnRate,"
         << "TargetX,TargetY,TargetType,RiskLevel,Damage_Potential,Time\n";

    for (unsigned int step = 0; step < steps; step++)
    {
        float time = step * dt;

        for (Threat& T : threats)
        {
            if (!T.active)
                continue;

            float measuredX = T.postion.x + noise(rng);
            float measuredY = T.postion.y + noise(rng);

            Vector2 measured_postion{measuredX, measuredY};

            const Target* targetInfo = findTargetInfo(targets, T.target.position);

            Vector2 Postion_Normalized = normalize(measured_postion);

            fout << T.type << ","
                 << T.ID << ","
                 << T.UAV_ID << ","
                 << T.postion.x << ","
                 << T.postion.y << ","
                 << measuredX << ","
                 << measuredY << ","
                 << Postion_Normalized.x << ","
                 << Postion_Normalized.y << ","
                 << T.velocity.x << ","
                 << T.velocity.y << ","
                 << T.direction.x << ","
                 << T.direction.y << ","
                 << T.speed << ","
                 << normalize_speed(T.speed) << ","
                 << T.heading << ","
                 << T.turn_gain << ","
                 << T.max_turn_rate << ","
                 << T.target.position.x << ","
                 << T.target.position.y << ",";

            unsigned int threatType = 0;

            if (strcmp(T.type, "Kalibr") == 0) threatType = KALIBR;

            if (strcmp(T.type, "Kinzhal") == 0) threatType = KINZHAL;

            int risk = getRiskLevel(T.UAV_ID, targetInfo->id);

            fout << targetInfo->id << ","
                << risk << ","
                << T.damage_potential << ","
                << time << "\n";
        }

        for (Threat& T : threats)
            T.update(dt);
    }

    fout.close();
}

int main()
{
    std::vector<Target> targets = {
        {{7500, 2000}, "Energy", 2},
        {{9000, 5000}, "Civilian", 1},
        {{2500, 4000}, "Landscape", 0}
    };

    std::vector<Threat> threats;

    unsigned int no_of_threats = 6;

    std::vector<int> desiredRiskCount = {10, 10, 10, 10, 10};
    std::vector<int> currentRiskCount(5, 0);

    while (threats.size() < no_of_threats)
    {
        Threat T = get_threat(static_cast<unsigned int>(threats.size() + 1), targets);

        const Target* targetInfo = findTargetInfo(targets, T.target.position);

        if (targetInfo == nullptr)
            continue;

        int risk = getRiskLevel(T.UAV_ID, targetInfo->id);

        if (currentRiskCount[risk] < desiredRiskCount[risk])
        {
            threats.push_back(T);
            currentRiskCount[risk]++;
        }
    }

    float dt = 0.1f;
    unsigned int steps = 1000;

    run_simulation(dt, steps, threats, targets);

    return 0;
}