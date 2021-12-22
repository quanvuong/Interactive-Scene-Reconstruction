#include "pg_map_ros/node_type.h"

#include <iostream>

using std::ostream;

namespace pgm
{

ostream & operator<<(ostream &os, const NodeType node_t)
{
    switch (node_t)
    {
    case NodeType::ObjectNode:
        os << "ObjectNode";
        break;

    case NodeType::ConceptNode:
        os << "ConceptNode";
        break;

    default:
        break;
    }

    return os;
}


ostream &operator<<(ostream& os, const Quaternion &quat)
{
    os << "(" << quat.x << ", " << quat.y << ", " << quat.z << ", " << quat.w << ")";

    return os;
}


ostream &operator<<(ostream& os, const Point &pt)
{
    os << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")";

    return os;
}


}  // end of namespace pgm
