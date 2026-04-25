#pragma once

#include <cstring>
#include <math.h>

#define PI 3.14159265359

#define SP_X1 3000
#define SP_X2 7000
#define SP_Y1 8000
#define SP_Y2 10000

enum Threat_Kind
{
    GERAN = 0,
    KALIBR = 1,
    KINZHAL = 2
};

struct Vector2
{
    float x, y;
};

float length(Vector2 v)
{ return sqrt(v.x * v.x + v.y * v.y); }

Vector2 normalize(Vector2 v)
{
    float len = length(v);

    if (len <= 0.0f)
        return {0.0f, 0.0f};

    return {v.x / len, v.y / len};
}

Vector2 operator+(Vector2 lhs, Vector2 rhs)
{ return {lhs.x + rhs.x, lhs.y + rhs.y}; }

Vector2 operator-(Vector2 lhs, Vector2 rhs)
{ return {lhs.x - rhs.x, lhs.y - rhs.y}; }

Vector2 operator*(Vector2 lhs, float k)
{ return {lhs.x * k, lhs.y * k}; }

float angle_to_target(Vector2 pos, Vector2 target)
{ return atan2(target.y - pos.y, target.x - pos.x); }

struct Target
{
    Vector2 position;
    const char* type;
    unsigned int id;
};

struct Threat
{
    unsigned int ID;
    unsigned int UAV_ID;
    char type[101];

    Vector2 postion;
    Vector2 velocity;
    Vector2 direction;
    Target target;

    float damage_potential;

    float speed;
    float heading;
    float turn_gain;
    float max_turn_rate;
    float hit_radius = 60.0f;
    
    bool left_map;
    bool active;
    bool reached_target;

    Threat(unsigned int ID, Vector2 start, Target target, float dmg_pot,
             float __speed, const char name[], unsigned int uav_id,
                float __turn_gain, float __max_turn_rate)
        : ID(ID), UAV_ID(uav_id), postion(start), target(target), damage_potential(dmg_pot),
             speed(__speed), turn_gain(__turn_gain), max_turn_rate(__max_turn_rate)
    {
        strcpy(type, name);

        heading = angle_to_target(start, target.position);
        init_velocity();

        left_map = false;
        active = true;
        reached_target = false;
    }

    void init_velocity()
    { 
        velocity.x = speed * std::cos(heading);
        velocity.y = speed * std::sin(heading);
        direction = normalize(velocity);
    }

    void update(float dt);
};

static float wrap_angle(float angle)
{
    while (angle > PI)
        angle -= 2.0f * PI; 

    while (angle < -PI)
        angle += 2.0f * PI;

    return angle;
}

static float clamp(float value, float min_value, float max_value)
{
    if (value < min_value)
        return min_value;

    if (value > max_value)
        return max_value;

    return value;
}

void Threat::update(float dt)
{
    if (!active) return;

    float angle = angle_to_target(postion, target.position);

    float error = wrap_angle(angle - heading);

    float heading_rate = turn_gain * error;

    heading_rate = clamp(heading_rate, -max_turn_rate, max_turn_rate);

    heading += heading_rate * dt;

    velocity.x = speed * cos(heading);
    velocity.y = speed * sin(heading);

    postion = postion + velocity * dt;

    if (length(target.position - postion) <= hit_radius)
    {
        active = false;
        reached_target = true;
    }

    if (postion.x < 0.0f || postion.x > 10000.0f || postion.y < 0.0f || postion.y > 10000.0f)
    {
        left_map = true;
        active = false;
    }
}