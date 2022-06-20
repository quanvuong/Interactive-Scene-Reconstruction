#ifndef _PGM_CONCEPT_NODE
#define _PGM_CONCEPT_NODE

#include <iostream>
#include <string>
#include <memory>

#include "node_type.h"
#include "node_base.h"


namespace pgm{

class ConceptNode: public NodeBase
{
public:
    using Ptr = std::shared_ptr<ConceptNode>;

    explicit ConceptNode(int, const std::string& = "");

    ~ConceptNode() = default;

    // overwrite the output function of the class
    friend std::ostream& operator<<(std::ostream&, const ConceptNode&);

    // friend std::ostream& operator<<(std::ostream&, const ConceptNode::Ptr);


/*******************************************************************
 * Modifier
 ******************************************************************/
public:
    void setConcept(const std::string &);


/*******************************************************************
 * Element Access
 ******************************************************************/
public:
    /**
     * Validate current parse graph
     *
     * Check if node_dict edge_set are correct. This method could
     * be time-consuming.
     *
     * @return true if valid, otherwise false
     */
    std::string getConcept() const;


protected:
    // Overwrite the pure virtual function from NodeBase
    std::ostream & output(std::ostream &) const override;

private:
    std::string concept_;
};

}  // end of namespace pgm

#endif
